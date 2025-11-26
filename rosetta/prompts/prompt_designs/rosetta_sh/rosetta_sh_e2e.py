import copy
import inspect

from rosetta.prompts.grounding import ground_preference
from rosetta.prompts.iterative_error_correction import o1mini_error_loop
from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils import *

def rosetta_sh_e2e(
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

    # PHASE 2: STAGING AND CODING 
    raw_env_code = inspect.getsource(ENV_ID_TO_SIM_CLS[env_id])
    env_code = prep_env_code(
        raw_env_code,
        act_space=act_space,
        simulator="maniskill",
        use_prior_reward=prev_funcs is not None
    )

    if prev_funcs is not None:
        env_code = replace_methods(env_code, prev_funcs)

    # Step 2: add code user message
    documentation = get_prompt_content(f"documentation/{act_space}")
    user_code_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_user"))
    user_code_msg.fill_dynamic_fields({
        "documentation": documentation,
        "environment_code": env_code,
        "grounded_preference": grounding_components["grounded_preference"],
        "demo_summary": grounding_components["summary"],
        "task_description": grounding_components["task_description"]
    })
    default_save_msg_hist(user_code_msg, hist, hist_f)
    default_save_msg_hist(user_code_msg, debug_hist, debug_f)

    # Step 3: run api, get preference code assistant message 
    asst_code_msg = query_until_complete(client, hist, "o1-mini", params)

    # Step 4: add preference code asst message to history and update function dict 
    default_save_msg_hist(asst_code_msg, hist, hist_f)
    default_save_msg_hist(asst_code_msg, debug_hist, debug_f)
    latest_funcs = update_latest_funcs(asst_code_msg, latest_funcs)
    all_funcs.append(latest_funcs.copy())

    # Step 11: error loop 
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

    print()
    print("Length of all funcs:", len(all_funcs))
    print() 

    return grounding_components, latest_funcs, all_funcs, num_stages