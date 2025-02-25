import copy
import inspect

from rosetta.prompts.grounding import ground_preference
from rosetta.prompts.iterative_error_correction import o1mini_error_loop
from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils import *


def rosetta_sh(
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

    # Phase 2: STAGING
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

    # Step 3: run api, get preference plan assistant message from gpt-4o
    asst_plan_msg = query_until_complete(client, hist, "gpt-4o", params)

    # Phase 2: CODING
    # Step 4: make alternative user stage reward message to give to o1-mini
    alt_user_plan_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fplanforcodestep_user"))
    alt_user_plan_msg.fill_dynamic_fields({
        "task_description": task_description,
        "demo_summary": grounding_components["summary"],
        "grounded_preference": grounding_components["grounded_preference"],
        "environment_code": env_code
    })
    hist[0] = alt_user_plan_msg
    hist[:] = [hist[0]]
    if hist_f is not None: 
        save_hist_to_json(hist, hist_f)
    default_save_msg_hist(alt_user_plan_msg, debug_hist, debug_f)

    # Step 5: add original assistant stage reward message to give to o1-mini 
    default_save_msg_hist(asst_plan_msg, hist, hist_f)
    default_save_msg_hist(asst_plan_msg, debug_hist, debug_f)

    # Step 6: add code user message
    documentation = get_prompt_content(f"documentation/{act_space}")
    user_code_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_user"))
    user_code_msg.fill_dynamic_fields({
        "documentation": documentation
    })
    default_save_msg_hist(user_code_msg, hist, hist_f)
    default_save_msg_hist(user_code_msg, debug_hist, debug_f)

    # Step 7: run api, get preference code assistant message 
    asst_code_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 8: add preference code asst message to history and update function dict 
    default_save_msg_hist(asst_code_msg, hist, hist_f)
    default_save_msg_hist(asst_code_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_code_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 9: check if latest_funcs is complete
    if set(latest_funcs.keys()) != set(funcs_to_overwrite):
        print("Completed Step 9: latest_funcs is incomplete, returning early")
        return grounding_components, latest_funcs, [], None

    # Step 10: first error loop 
    tmp_hist = copy.deepcopy(hist)
    if ("skip_error_testing" not in kwargs) or (not kwargs["skip_error_testing"]):
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

    asst_corrected_code_content = replace_methods(asst_code_msg.content, latest_funcs)
    asst_corrected_code_msg = PromptMessage(role="assistant", content=asst_corrected_code_content)
    hist[-1] = asst_corrected_code_msg

    # Phase 3: VERIFICATION QUESTIONS
    # Step 11: add geometry review user message to history 
    user_geom_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_geomcot_user"))
    default_save_msg_hist(user_geom_msg, hist, hist_f)
    default_save_msg_hist(user_geom_msg, debug_hist, debug_f)

    # Step 12: get geometry-reviewed code assistant message from o1-mini 
    asst_geom_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 13: add geometry-reviewed code asst message to history and update function dict
    default_save_msg_hist(asst_geom_msg, hist, hist_f)
    default_save_msg_hist(asst_geom_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_geom_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 14: add target positions review user message to history 
    user_targets_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_targetscot_user"))
    default_save_msg_hist(user_targets_msg, hist, hist_f)
    default_save_msg_hist(user_targets_msg, debug_hist, debug_f)

    # Step 15: get targets-reviewed code assistant message from o1-mini 
    asst_targets_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 16: add targets-reviewed code asst message to history and update function dict
    default_save_msg_hist(asst_targets_msg, hist, hist_f)
    default_save_msg_hist(asst_targets_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_targets_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 17: add density review user message to history 
    user_dense_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_densecot_user"))
    default_save_msg_hist(user_dense_msg, hist, hist_f)
    default_save_msg_hist(user_dense_msg, debug_hist, debug_f)

    # Step 18: get density-reviewed code assistant message from o1-mini 
    asst_dense_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 19: add density-reviewed code asst message to history and update function dict
    default_save_msg_hist(asst_dense_msg, hist, hist_f)
    default_save_msg_hist(asst_dense_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_dense_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 20: add masking review user message to history
    user_mask_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_maskingcot_user"))
    default_save_msg_hist(user_mask_msg, hist, hist_f)
    default_save_msg_hist(user_mask_msg, debug_hist, debug_f)

    # Step 21: get masking-reviewed code assistant message from o1-mini
    asst_mask_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 22: add masking-reviewed code asst message to history and update function dict
    default_save_msg_hist(asst_mask_msg, hist, hist_f)
    default_save_msg_hist(asst_mask_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_mask_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 23: loop error correction
    if ("skip_error_testing" not in kwargs) or (not kwargs["skip_error_testing"]):
        latest_funcs = o1mini_error_loop(
            client,
            params,
            NUM_ERROR_CORR_TRIES,
            latest_funcs,
            env_id, 
            funcs_to_overwrite,
            act_space,
            env_code,
            hist,
            # hist_f,
            debug_hist,
            debug_f
        )
        all_funcs.append(latest_funcs.copy())

    print()
    print("Length of all_funcs:", len(all_funcs))
    print()

    return grounding_components, latest_funcs, all_funcs, num_stages
