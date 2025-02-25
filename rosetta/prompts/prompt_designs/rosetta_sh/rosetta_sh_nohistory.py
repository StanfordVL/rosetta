import inspect

from rosetta.prompts.grounding import ground_preference
from rosetta.prompts.iterative_error_correction import o1mini_error_loop
from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils import *


def rosetta_sh_nohistory(
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
    funcs_to_overwrite = FUNCS_TO_OVERWRITE[act_space]
    documentation = get_prompt_content(f"documentation/{act_space}")
    env_code_no_reward = replace_methods(env_code, {"evaluate": "", "compute_dense_reward": ""})

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

    # Phase 2: gpt-4o plan 

    # Step 2: add preference CoT reward system message to history 
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
        "environment_code": env_code
    })
    default_save_msg_hist(user_plan_msg, hist, hist_f)
    default_save_msg_hist(user_plan_msg, debug_hist, debug_f)

    # Step 3: run api, get preference plan asst message from gpt-4o
    asst_plan_msg = query_until_complete(client, hist, "gpt-4o", params)

    # Step 4: get plan message out
    plan = extract_plan(asst_plan_msg)

    # Phase 2 o1-mini code draft 

    # Step 5: clear history and add code user message 
    user_code_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_user"))
    user_code_msg.fill_dynamic_fields({
        "documentation": documentation,
        "plan": plan,
        "environment_code": env_code,
        "grounded_preference": grounding_components["grounded_preference"]
    })
    hist = [user_code_msg]
    if hist_f is not None: 
        save_hist_to_json(hist, hist_f)
    default_save_msg_hist(user_code_msg, debug_hist, debug_f)

    # Step 6: run api, get preference code assistant message 
    asst_code_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 7: update latest funcs 
    latest_funcs = update_latest_funcs(asst_code_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 8: check if latest_funcs is complete 
    if set(latest_funcs.keys()) != set(funcs_to_overwrite):
        return grounding_components, latest_funcs, [], None 
    
    # Phase 3: first error loop 
    tmp_hist = []
    latest_funcs = o1mini_error_loop(
        client,
        params,
        NUM_ERROR_CORR_TRIES,
        latest_funcs,
        env_id,
        funcs_to_overwrite,
        act_space,
        env_code,
        tmp_hist,
        debug_hist,
        debug_f
    )
    all_funcs.append(latest_funcs.copy())

    # Phase 4: review questions (separate, not CoT)

    # Step 9: empty history and add geometry review user message
    user_geom_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_geomcot_user"))
    user_geom_msg.fill_dynamic_fields({
        "evaluate": latest_funcs["evaluate"],
        "compute_dense_reward": latest_funcs["compute_dense_reward"],
        "env_code_no_reward": env_code_no_reward,
        "documentation": documentation
    })
    hist = []
    default_save_msg_hist(user_geom_msg, hist, hist_f)
    default_save_msg_hist(user_geom_msg, debug_hist, debug_f)

    # Step 10: get geometry-reviewed code asst message from o1-mini and update function dict
    asst_geom_msg = query_until_complete(client, hist, "o1-mini", params)
    default_save_msg_hist(asst_geom_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_geom_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 11: empty history and add target positions review user message
    user_targets_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_targetscot_user"))
    user_targets_msg.fill_dynamic_fields({
        "evaluate": latest_funcs["evaluate"],
        "compute_dense_reward": latest_funcs["compute_dense_reward"],
        "env_code_no_reward": env_code_no_reward,
        "documentation": documentation
    })
    hist = []
    default_save_msg_hist(user_targets_msg, hist, hist_f)
    default_save_msg_hist(user_targets_msg, debug_hist, debug_f)

    # Step 12: get targets-reviewed code asst message from o1-mini and update function dict
    asst_targets_msg = query_until_complete(client, hist, "o1-mini", params)
    default_save_msg_hist(asst_targets_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_targets_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 13: empty history and add density review user message 
    user_dense_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_densecot_user"))
    user_dense_msg.fill_dynamic_fields({
        "evaluate": latest_funcs["evaluate"],
        "compute_dense_reward": latest_funcs["compute_dense_reward"],
        "env_code_no_reward": env_code_no_reward,
        "documentation": documentation
    })
    hist = []
    default_save_msg_hist(user_targets_msg, hist, hist_f)
    default_save_msg_hist(user_dense_msg, debug_hist, debug_f)

    # Step 14: get density-reviewed code asst message from o1-mini and update function dict
    asst_dense_msg = query_until_complete(client, hist, "o1-mini", params)
    default_save_msg_hist(asst_dense_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_dense_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 15: empty history and add masking review user message 
    user_masking_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_maskingcot_user"))
    user_masking_msg.fill_dynamic_fields({
        "evaluate": latest_funcs["evaluate"],
        "compute_dense_reward": latest_funcs["compute_dense_reward"],
        "env_code_no_reward": env_code_no_reward,
        "documentation": documentation
    })
    hist = []
    default_save_msg_hist(user_targets_msg, hist, hist_f)
    default_save_msg_hist(user_masking_msg, debug_hist, debug_f)

    # Step 16: get masking-reviewed code asst message from o1-mini and update function dict
    asst_masking_msg = query_until_complete(client, hist, "o1-mini", params)
    default_save_msg_hist(asst_masking_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_masking_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Phase 5: second error loop
    tmp_hist = []
    latest_funcs = o1mini_error_loop(
        client,
        params,
        NUM_ERROR_CORR_TRIES,
        latest_funcs,
        env_id,
        funcs_to_overwrite,
        act_space,
        env_code,
        tmp_hist,
        debug_hist,
        debug_f
    )
    all_funcs.append(latest_funcs.copy())

    print()
    print()
    print("Length all funcs:", len(all_funcs))
    print()
    print()

    # Final functions, list of all functions, num_stages (None, only for compatibility with action primitive prompt design function output signatures)
    return grounding_components, latest_funcs, all_funcs, None