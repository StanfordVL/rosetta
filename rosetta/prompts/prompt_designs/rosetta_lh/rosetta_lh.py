import inspect

from rosetta.prompts.grounding import ground_preference
from rosetta.prompts.iterative_error_correction import o1mini_error_loop
from rosetta.prompts.prompt_content.grounding.env_specific import ENV_ID_TO_GROUNDING_CLS
from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils import *


def rosetta_lh(
    human_input,
    env_id,
    prev_funcs,
    act_space,
    task_description,
    demo_dir,
    client,
    content_version,
    hist,
    hist_f,
    debug_hist,
    debug_f,
    params,
    **kwargs
):
    all_funcs = []
    latest_funcs = {}
    latest_interim_funcs = {}
    num_stages = None 
    funcs_to_overwrite = FUNCS_TO_OVERWRITE[act_space]

    # PHASE 1: GROUNDING 
    # Step 1: ground preference 
    grounding_components = ground_preference(
        demo_dir,
        human_input,
        env_id,
        act_space,
        client,
        task_description
    )    
    env_info = ENV_ID_TO_GROUNDING_CLS[env_id]()

    # Phase 1: gpt-4o plan 
    # Step 1: add preference CoT reward system message to history
    system_msg = PromptMessage(role="system", content=get_prompt_content(f"{content_version}/fr_system"))
    default_save_msg_hist(system_msg, hist, hist_f)
    default_save_msg_hist(system_msg, debug_hist, debug_f)

    # Step 2: add preference plan user message to history
    # Prepare environment code
    raw_env_code = inspect.getsource(ENV_ID_TO_SIM_CLS[env_id])
    env_code = prep_env_code(
        raw_env_code,
        act_space=act_space,
        simulator="maniskill",
        use_prior_reward=prev_funcs is not None
    )
    if prev_funcs is not None:
        env_code = replace_methods(env_code, prev_funcs)

    user_plan_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fplan_user"))
    user_plan_msg.fill_dynamic_fields({
        "task_description": grounding_components["task_description"],
        "demo_summary": grounding_components["summary"],
        "grounded_preference": grounding_components["grounded_preference"],
        "environment_code": env_code,
        "new_description": grounding_components["next_description"],
        "setup_description": env_info.setup_description
    })
    default_save_msg_hist(user_plan_msg, hist, hist_f)
    default_save_msg_hist(user_plan_msg, debug_hist, debug_f)

    # Step 3: run api, get preference plan assistant message from gpt-4o
    asst_plan_msg = query_until_complete(client, hist, "gpt-4o", params)

    # Step 4: get target action list from preference plan assistant message
    target_actions = extract_target_actions(asst_plan_msg)

    # Phase 2: o1-mini code draft
    # Step 5: make alternative user stage reward message to give to o1-mini
    alt_user_plan_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fplanforcodestep_user"))
    alt_user_plan_msg.fill_dynamic_fields({
        "task_description": grounding_components["task_description"],
        "demo_summary": grounding_components["summary"],
        "grounded_preference": grounding_components["grounded_preference"],
        "environment_code": env_code,
        "new_description": grounding_components["next_description"],
        "setup_description": env_info.setup_description
    })
    hist[0] = alt_user_plan_msg
    hist[:] = [hist[0]]
    if hist_f is not None: 
        save_hist_to_json(hist, hist_f)
    default_save_msg_hist(alt_user_plan_msg, debug_hist, debug_f)

    # Step 6: add original assistant stage reward message to give to o1-mini 
    default_save_msg_hist(asst_plan_msg, hist, hist_f)
    default_save_msg_hist(asst_plan_msg, debug_hist, debug_f)

    # Step 7: add code user message
    user_code_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_user"))
    user_code_msg.fill_dynamic_fields({
        "setup_description": env_info.setup_description,
        "info_keys": env_info.info_keys,
        "task_description": grounding_components["task_description"],
        "grounded_preference": grounding_components["grounded_preference"],
        "demo_summary": grounding_components["summary"],
        "new_description": grounding_components["next_description"]
    })

    default_save_msg_hist(user_code_msg, hist, hist_f)
    default_save_msg_hist(user_code_msg, debug_hist, debug_f)

    # Step 8: run api, get preference code assistant message 
    asst_code_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 9: add preference code asst message to history and update function dict 
    default_save_msg_hist(asst_code_msg, hist, hist_f)
    default_save_msg_hist(asst_code_msg, debug_hist, debug_f)

    latest_interim_funcs = update_latest_interim_funcs_actprim(asst_code_msg, latest_interim_funcs, target_actions)
    latest_funcs = build_actprim_funcs(latest_interim_funcs, target_actions)
    all_funcs.append(latest_funcs.copy())

    # Step 10: if latest_funcs is incomplete, return early to avoid doing review steps on nothing and wasting tokens/potentially inducing code inappropriately
    if set(latest_funcs.keys()) != set(funcs_to_overwrite):
        return grounding_components, latest_funcs, [], num_stages

    # Phase 3: CoT of review questions
    # Step 11: add geometry review user message to history 
    user_geom_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_geometrycot_user"))
    user_geom_msg.fill_dynamic_fields({
        "setup_description": env_info.setup_description
    })
    default_save_msg_hist(user_geom_msg, hist, hist_f)
    default_save_msg_hist(user_geom_msg, debug_hist, debug_f)

    # Step 12: get geometry-reviewed code assistant message from o1-mini 
    asst_geom_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 13: add geometry-reviewed code asst message to history and update function dict
    default_save_msg_hist(asst_geom_msg, hist, hist_f)
    default_save_msg_hist(asst_geom_msg, debug_hist, debug_f)

    latest_interim_funcs = update_latest_interim_funcs_actprim(asst_geom_msg, latest_interim_funcs, target_actions)
    latest_funcs = build_actprim_funcs(latest_interim_funcs, target_actions)
    all_funcs.append(latest_funcs.copy())

    # Step 14: add normalization review user message to history
    user_norm_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_normcot_user"))
    default_save_msg_hist(user_norm_msg, hist, hist_f)
    default_save_msg_hist(user_norm_msg, debug_hist, debug_f)

    # Step 15: get normalization-reviewed code assistant message from o1-mini
    asst_norm_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 16: add normalization-reviewed code asst message to history and update function dict
    default_save_msg_hist(asst_norm_msg, hist, hist_f)
    default_save_msg_hist(asst_norm_msg, debug_hist, debug_f)

    latest_interim_funcs = update_latest_interim_funcs_actprim(asst_norm_msg, latest_interim_funcs, target_actions)
    latest_funcs = build_actprim_funcs(latest_interim_funcs, target_actions)
    all_funcs.append(latest_funcs.copy())

    # Step 17: add code cleanup review user message to history 
    user_clean_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_codecleanupcot_user"))
    default_save_msg_hist(user_clean_msg, hist, hist_f)
    default_save_msg_hist(user_clean_msg, debug_hist, debug_f)
    
    # Step 18: get code cleanup-reviewed code assistant message from o1-mini
    asst_clean_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 19: add code cleanup-reviewed code asst message to history and update function dict
    default_save_msg_hist(asst_clean_msg, hist, hist_f)
    default_save_msg_hist(asst_clean_msg, debug_hist, debug_f)

    latest_interim_funcs = update_latest_interim_funcs_actprim(asst_clean_msg, latest_interim_funcs, target_actions)
    latest_funcs = build_actprim_funcs(latest_interim_funcs, target_actions)
    all_funcs.append(latest_funcs.copy())

    # Phase 4: error correction 
    # Step 20: loop error correction
    print("Entering error loop")
    num_stages = extract_num_stages(latest_interim_funcs)
    if ("skip_error_testing" not in kwargs) or (not kwargs["skip_error_testing"]):
        latest_funcs = o1mini_error_loop(
            client,
            params,
            NUM_ERROR_CORR_TRIES,
            latest_funcs,
            env_id,
            funcs_to_overwrite,
            "actprim",
            env_code,
            hist,
            debug_hist,
            debug_f,
            stages=list(range(num_stages)),
            target_actions=target_actions
        )
        all_funcs.append(latest_funcs.copy())
    print("Done with error correction")

    return grounding_components, latest_funcs, all_funcs, num_stages