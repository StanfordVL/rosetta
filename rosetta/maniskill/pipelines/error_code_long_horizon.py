import multiprocessing
import fire
from copy import deepcopy


def run_in_separate_process(env_id=None, func_dict=None, json_path=None, functions_to_overwrite=None, stage=0, target_action=0):
    try:
        import rosetta.maniskill.long_env
        import rosetta.maniskill.short_env
        import numpy as np
        import gymnasium as gym
        import traceback
        from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
        from rosetta.maniskill.wrappers.skill_wrapper import SkillGymWrapper
        from rosetta.maniskill.pipelines.reward_manipulator import RewardManipulator
        if functions_to_overwrite is None:
            functions_to_overwrite = ['skill_reward', 'evaluate']
        rm = RewardManipulator(env_id=env_id, json_path=json_path, func_dict=func_dict)
        env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array", sim_backend="cpu", stage=stage)
        env = gym.make(rm.env_id, num_envs=1, enable_shadow=True, **env_kwargs)
        env = CPUGymWrapper(env)
        env = SkillGymWrapper(env,
                              skill_indices=env.task_skill_indices,
                              max_episode_steps=3,
                              max_steps_per_video=1,)
        og_function_dict = rm.get_class_methods_source(rm.env_cls)
        diff_functions = rm.get_diff_functions(rm.new_method_dict, og_function_dict)
        rm.change_function(functions_to_overwrite)
        obs, _ = env.reset()
        success = True
        env.cur_stage = stage

        # Ensure target_action is a valid action
        if target_action not in env.unwrapped.task_skill_indices.keys():
            return {'success': False, 'diff_functions': diff_functions, 'error_message': f'Target action {target_action} for stage {stage} is invalid. Check the selected stage actions in the original plan.'}

        # for i in range(10):
        try:
            stub_predicted_action = env.action_space.sample()
            stub_predicted_action[:len(env.unwrapped.task_skill_indices.keys())] = 0.
            stub_predicted_action[target_action] = 1.
            obs, reward, terminated, truncated, info = env.step(stub_predicted_action)
            
            prev_info=env.prev_info
            cur_info=env.cur_info
            
            # Generated code only exists where prev_info[f"stage{stage}_success"] = True and predicted_action == target_action
            # So, just make that boolean setting and action setting.
            # prev_info[f"stage{stage}_success"] = True # Jerry: doesn't apply to new framework i think
            reward=env.unwrapped.skill_reward(prev_info, cur_info, stub_predicted_action)
            # print(reward)
            reward = sum(reward.values())
            
            if not np.isscalar(reward):
                print("Reward is not a scalar")
                print(reward)
            assert np.isscalar(reward)
        except Exception as e:
            detailed_error_message = traceback.format_exc()
            success = False
        
        if success:
            return {'success': True, 'diff_functions': diff_functions}
        else:
            return {'success': False, 'diff_functions': diff_functions, 'error_message': detailed_error_message}
        env.close()
    except Exception as e:
        detailed_error_message = traceback.format_exc()
        return {'success': False, 'error_message': detailed_error_message}


def direct_evaluation_run_code(env_id=None, func_dict=None, json_path=None, functions_to_overwrite=None, stages=None, target_actions=None):
    """
    Runs simulations for multiple stages sequentially.

    Returns a list of result dictionaries.
    """
    if stages is None:
        stages = [0]  # Default to stage 0 if not provided

    if not isinstance(stages, list):
        # logging.error("Stages should be a list of integers.")
        print("Stages should be a list of integers.")
        sys.exit(1)

    # results = []

    for stage, target_action in zip(stages, target_actions):
        result_queue = multiprocessing.Queue()

        # Define the target function with all necessary arguments
        def target(q, env_id, func_dict, json_path, functions_to_overwrite, stage, target_action):
            result = run_in_separate_process(
                env_id=env_id,
                func_dict=func_dict,
                json_path=json_path,
                functions_to_overwrite=functions_to_overwrite,
                stage=stage,
                target_action=target_action
            )
            q.put(result)

        process = multiprocessing.Process(
            target=target,
            args=(result_queue, env_id, func_dict, json_path, functions_to_overwrite, stage, target_action)
        )

        try:
            print(f"Starting simulation for stage {stage} with target_action {target_action}...")
            process.start()
            
            # Optionally, set a timeout (e.g., 300 seconds)
            process.join(timeout=900)

            if process.is_alive():
                print(f"Process for stage {stage} is taking too long. Terminating...")
                process.terminate()
                process.join()
                result = {
                    'stage': stage,
                    'success': False,
                    'error_message': 'Process terminated due to timeout.'
                }
                continue

            if not result_queue.empty():
                result = result_queue.get()
                # results.append(result)
                if result.get('success'):
                    print(f"Stage {stage}: Success")
                else:
                    error_message = result.get('error_message', 'Unknown Error')
                    print(f"Stage {stage}: Failed with error:\n{error_message}\n")
                    break
            else:
                print(f"No result returned for stage {stage}.")
                result = {
                    'stage': stage,
                    'success': False,
                    'error_message': 'No result returned.'
                }

        except Exception as e:
            print(f"An exception occurred while running stage {stage}: {e}")
            detailed_error_message = traceback.format_exc()
            result = {
                'stage': stage,
                'success': False,
                'error_message': detailed_error_message
            }
        finally:
            if process.is_alive():
                print(f"Ensuring termination of process for stage {stage}...")
                process.terminate()
                process.join()

    return result

if __name__ == '__main__':
    # usage: python direct_evaluation_run_code.py --json_path "example0.txt"
    # or: python direct_evaluation_run_code.py --text "def evaluate():..."
    fire.Fire(direct_evaluation_run_code)