import copy
import inspect

from feedback_to_reward_prompts.grounding import ground_feedback
from feedback_to_reward_prompts.iterative_error_correction import o1mini_error_loop
from feedback_to_reward_prompts.prompt_message import PromptMessage
from feedback_to_reward_prompts.utils import *


def no_follow_up(
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
    # Step 1: ground feedback 
    grounding_components = ground_feedback(
        demo_dir,
        human_input,
        env_id,
        act_space,
        client,
        task_description
    )

    # Phase 1: gpt-4o plan

    # Step 1: add feedback CoT reward system message to history 
    print("Starting Step 1: Add feedback CoT reward system message to history")
    system_msg = PromptMessage(role="system", content=get_prompt_content(f"{content_version}/fr_system"))
    default_save_msg_hist(system_msg, hist, hist_f)
    default_save_msg_hist(system_msg, debug_hist, debug_f)
    print("Completed Step 1")

    # Step 2: add feedback plan user message to history 
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
        "demo_summary": grounding_components["demo_summary"],
        "grounded_feedback": grounding_components["grounded_feedback"],
        "environment_code": env_code
    })
    default_save_msg_hist(user_plan_msg, hist, hist_f)
    default_save_msg_hist(user_plan_msg, debug_hist, debug_f)
    print("Completed Step 2")

    # Step 3: run api, get feedback plan assistant message from gpt-4o
    print("Starting Step 3: Run API to get feedback plan assistant message from gpt-4o")
    asst_plan_msg = query_until_complete(client, hist, "gpt-4o", params)
    print("Completed Step 3")

    # Phase 2: o1-mini code draft

    # Step 4: make alternative user stage reward message to give to o1-mini
    print("Starting Step 4: Make alternative user stage reward message for o1-mini")
    alt_user_plan_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fplanforcodestep_user"))
    alt_user_plan_msg.fill_dynamic_fields({
        "task_description": task_description,
        "demo_summary": grounding_components["demo_summary"],
        "grounded_feedback": grounding_components["grounded_feedback"],
        "environment_code": env_code
    })
    hist[0] = alt_user_plan_msg
    hist[:] = [hist[0]]
    if hist_f is not None: 
        save_hist_to_json(hist, hist_f)
    default_save_msg_hist(alt_user_plan_msg, debug_hist, debug_f)
    print("Completed Step 4")

    # Step 5: add original assistant stage reward message to give to o1-mini 
    print("Starting Step 5: Add original assistant stage reward message to history")
    default_save_msg_hist(asst_plan_msg, hist, hist_f)
    default_save_msg_hist(asst_plan_msg, debug_hist, debug_f)
    print("Completed Step 5")

    # Step 6: add code user message
    print("Starting Step 6: Add code user message to history")
    documentation = get_prompt_content(f"documentation/{act_space}")
    user_code_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_user"))
    user_code_msg.fill_dynamic_fields({
        "documentation": documentation
    })
    default_save_msg_hist(user_code_msg, hist, hist_f)
    default_save_msg_hist(user_code_msg, debug_hist, debug_f)
    print("Completed Step 6")

    # Step 7: run api, get feedback code assistant message 
    print("Starting Step 7: Run API to get feedback code assistant message from o1-mini")
    asst_code_msg = query_until_complete(client, hist, "o1-mini", params)
    print("Completed Step 7")

    # Step 8: add feedback code asst message to history and update function dict 
    print("Starting Step 8: Add feedback code assistant message to history and update functions")
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

    return {"feedback": human_input}, latest_funcs, all_funcs, num_stages

