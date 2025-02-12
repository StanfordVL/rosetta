import copy
import inspect

from rosetta.prompts.grounding import ground_preference
from rosetta.prompts.iterative_error_correction import o1mini_error_loop
from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils import *


def no_staging(
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
    
    # Step 6: add code user message
    print("Starting Step 6: Add code user message to history")
    documentation = get_prompt_content(f"documentation/{act_space}")
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

    user_code_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_user"))
    user_code_msg.fill_dynamic_fields({
        "documentation": documentation,
        "task_description": grounding_components["task_description"],
        "demo_summary": grounding_components["demo_summary"],
        "grounded_preference": grounding_components["grounded_preference"],
        "environment_code": env_code 
    })
    default_save_msg_hist(user_code_msg, hist, hist_f)
    default_save_msg_hist(user_code_msg, debug_hist, debug_f)
    print("Completed Step 6")

    # Step 7: run api, get preference code assistant message 
    print("Starting Step 7: Run API to get preference code assistant message from o1-mini")
    asst_code_msg = query_until_complete(client, hist, "o1-mini", params)
    print("Completed Step 7")

    # Step 8: add preference code asst message to history and update function dict 
    print("Starting Step 8: Add preference code assistant message to history and update functions")
    default_save_msg_hist(asst_code_msg, hist, hist_f)
    default_save_msg_hist(asst_code_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_code_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())
    print("Completed Step 8")

    # Step 9: check if latest_funcs is complete
    print("Starting Step 9: Check if latest_funcs is complete")
    if set(latest_funcs.keys()) != set(funcs_to_overwrite):
        print("Completed Step 9: latest_funcs is incomplete, returning early")
        return grounding_components, latest_funcs, [], None
    print("Completed Step 9: latest_funcs is complete")

    # Phase 3: first error loop
    print("Starting Step 10: First error loop")
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
    print("Completed Step 10: First error loop")

    print("Starting step 11: Replace latest message with error-corrected code")
    asst_corrected_code_content = replace_methods(asst_code_msg.content, latest_funcs)
    asst_corrected_code_msg = PromptMessage(role="assistant", content=asst_corrected_code_content)
    hist[-1] = asst_corrected_code_msg
    print("Completed Step 11: error code is corrected")

    # Phase 4: CoT of review questions

    # Step 10: add geometry review user message to history 
    print("Starting Step 10: Add geometry review user message to history")
    user_geom_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_geomcot_user"))
    default_save_msg_hist(user_geom_msg, hist, hist_f)
    default_save_msg_hist(user_geom_msg, debug_hist, debug_f)
    print("Completed Step 10")

    # Step 11: get geometry-reviewed code assistant message from o1-mini 
    print("Starting Step 11: Get geometry-reviewed code assistant message from o1-mini")
    asst_geom_msg = query_until_complete(client, hist, "o1-mini", params)
    print("Completed Step 11")

    # Step 12: add geometry-reviewed code asst message to history and update function dict
    print("Starting Step 12: Add geometry-reviewed code assistant message to history and update functions")
    default_save_msg_hist(asst_geom_msg, hist, hist_f)
    default_save_msg_hist(asst_geom_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_geom_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())
    print("Completed Step 12")

    # Step 13: add target positions review user message to history 
    print("Starting Step 13: Add target positions review user message to history")
    user_targets_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_targetscot_user"))
    default_save_msg_hist(user_targets_msg, hist, hist_f)
    default_save_msg_hist(user_targets_msg, debug_hist, debug_f)
    print("Completed Step 13")

    # Step 14: get targets-reviewed code assistant message from o1-mini 
    print("Starting Step 14: Get targets-reviewed code assistant message from o1-mini")
    asst_targets_msg = query_until_complete(client, hist, "o1-mini", params)
    print("Completed Step 14")

    # Step 15: add targets-reviewed code asst message to history and update function dict
    print("Starting Step 15: Add targets-reviewed code assistant message to history and update functions")
    default_save_msg_hist(asst_targets_msg, hist, hist_f)
    default_save_msg_hist(asst_targets_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_targets_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())
    print("Completed Step 15")

    # Step 16: add density review user message to history 
    print("Starting Step 16: Add density review user message to history")
    user_dense_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_densecot_user"))
    default_save_msg_hist(user_dense_msg, hist, hist_f)
    default_save_msg_hist(user_dense_msg, debug_hist, debug_f)
    print("Completed Step 16")

    # Step 17: get density-reviewed code assistant message from o1-mini 
    print("Starting Step 17: Get density-reviewed code assistant message from o1-mini")
    asst_dense_msg = query_until_complete(client, hist, "o1-mini", params)
    print("Completed Step 17")

    # Step 18: add density-reviewed code asst message to history and update function dict
    print("Starting Step 18: Add density-reviewed code assistant message to history and update functions")
    default_save_msg_hist(asst_dense_msg, hist, hist_f)
    default_save_msg_hist(asst_dense_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_dense_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())
    print("Completed Step 18")

    # Step 19: add masking review user message to history
    print("Starting Step 19: Add masking review user message to history")
    user_mask_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_maskingcot_user"))
    default_save_msg_hist(user_mask_msg, hist, hist_f)
    default_save_msg_hist(user_mask_msg, debug_hist, debug_f)
    print("Completed Step 19")

    # Step 20: get masking-reviewed code assistant message from o1-mini
    print("Starting Step 20: Get masking-reviewed code assistant message from o1-mini")
    asst_mask_msg = query_until_complete(client, hist, "o1-mini", params)
    print("Completed Step 20")

    # Step 21: add masking-reviewed code asst message to history and update function dict
    print("Starting Step 21: Add masking-reviewed code assistant message to history and update functions")
    default_save_msg_hist(asst_mask_msg, hist, hist_f)
    default_save_msg_hist(asst_mask_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_mask_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())
    print("Completed Step 21")

    # Phase 4: error correction

    # Step 22: loop error correction
    print("Starting Step 22: Entering error correction loop using o1-mini as error corrector")
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
    print("Completed Step 22")

    print()
    print("Length of all_funcs:", len(all_funcs))
    print()

    return grounding_components, latest_funcs, all_funcs, num_stages

