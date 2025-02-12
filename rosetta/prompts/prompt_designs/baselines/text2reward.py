import copy
import inspect

from rosetta.prompts.iterative_error_correction import o1mini_error_loop_ro
from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils import *

    
def text2reward(
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
    funcs_to_overwrite = ["compute_dense_reward"]

    intro_user_msg=PromptMessage(role="user", content=get_prompt_content(f"{content_version}/intro_user"))
    default_save_msg_hist(intro_user_msg, hist, hist_f)
    default_save_msg_hist(intro_user_msg, debug_hist, debug_f)

    user_code_msg = PromptMessage(role="user", content=get_prompt_content(f"{content_version}/fcode_user"))
    user_code_msg.fill_dynamic_fields({
        "environment_desc": get_prompt_content.get(f"{content_version}/env_specific/{env_id}"),
        "env_class_desc": get_prompt_content.get(f"{content_version}/env_class_desc"),
        "documentation": get_prompt_content(f"documentation/{act_space}"),
        "compute_dense_reward": prev_funcs.get("compute_dense_reward", None),
        "original_feedback": human_input
    })
    default_save_msg_hist(user_code_msg, hist, hist_f)
    default_save_msg_hist(user_code_msg, debug_hist, debug_f)

    asst_code_msg=query_until_complete(client, hist, "o1-mini", params)
    default_save_msg_hist(asst_code_msg, hist, hist_f)
    default_save_msg_hist(asst_code_msg, debug_hist, debug_f)

    latest_funcs = update_latest_funcs(asst_code_msg, latest_funcs)
    
    # Iterative error correction
    tmp_hist = copy.deepcopy(hist)
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
    if ("skip_error_testing" not in kwargs) or (not kwargs["skip_error_testing"]):
        latest_funcs = o1mini_error_loop_ro(
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
    
    return {"feedback": human_input}, latest_funcs, all_funcs, num_stages