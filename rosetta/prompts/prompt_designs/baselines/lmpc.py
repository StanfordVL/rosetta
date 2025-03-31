import copy
import inspect
import traceback

from rosetta.prompts.iterative_error_correction import o1mini_error_loop_ro
from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils import *


def lmpc(
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
    num_stages = None
    funcs_to_overwrite = ["compute_dense_reward"]
    inner_num_gen = kwargs.get("inner_num_gen", 1)
    
    i=0
    error=0
    while i<inner_num_gen:
        try:
            print(f"Generation {i} for the lmpc starts")
            tmp_hist = []
            tmp_latest_funcs={}

            intro_user_msg=PromptMessage(role="user", content=get_prompt_content(f"{content_version}/intro_user"))
            default_save_msg_hist(intro_user_msg, tmp_hist, hist_f)
            default_save_msg_hist(intro_user_msg, debug_hist, debug_f)
            
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
                "environment_code": env_code,
                "compute_dense_reward": prev_funcs.get("compute_dense_reward", None),
                "human_feedback": human_input,
            })
            
            default_save_msg_hist(user_code_msg, tmp_hist, hist_f)
            default_save_msg_hist(user_code_msg, debug_hist, debug_f)
        
            reward_code_msg=query_until_complete(client, tmp_hist, "o1-mini", params)
            default_save_msg_hist(reward_code_msg, tmp_hist, hist_f)
            default_save_msg_hist(reward_code_msg, debug_hist, debug_f)
            tmp_latest_funcs = update_latest_funcs(reward_code_msg, tmp_latest_funcs)
            
            tmp_hist_hist = copy.deepcopy(tmp_hist)
            if ("skip_error_testing" not in kwargs) or (not kwargs["skip_error_testing"]):
                tmp_latest_funcs = o1mini_error_loop_ro(
                    client, 
                    params, 
                    NUM_ERROR_CORR_TRIES,
                    tmp_latest_funcs,
                    env_id,
                    funcs_to_overwrite,
                    act_space, 
                    env_code,
                    tmp_hist_hist,
                    debug_hist,
                    debug_f
                )
            all_funcs.append(tmp_latest_funcs)
            i+=1
        except Exception as e:
            print("Error in lmpc generation ",i)
            print(traceback.format_exc())
            error+=1
        if error>3:
            raise Exception(f"Too many errors ({error}) in the generation")
    
    return {"preference": human_input}, tmp_latest_funcs, all_funcs, num_stages
