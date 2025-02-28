import multiprocessing
import fire

def run_in_separate_process(env_id=None, func_dict=None, json_path=None, num_envs=4, sim_backend="gpu", functions_to_overwrite=None):    
    try:
        import gymnasium as gym
        import traceback
        from rosetta.maniskill.pipelines.reward_manipulator import RewardManipulator
        if functions_to_overwrite is None:
            functions_to_overwrite = ['evaluate', 'compute_dense_reward', 'compute_normalized_dense_reward']
        rm = RewardManipulator(env_id=env_id, json_path=json_path, func_dict=func_dict,sim_backend=sim_backend)
        env = gym.make(rm.env_id, num_envs=num_envs, sim_backend=sim_backend)
        og_function_dict = rm.get_class_methods_source(rm.env_cls)
        diff_functions = rm.get_diff_functions(rm.new_method_dict, og_function_dict)
        rm.change_function(functions_to_overwrite)
        obs, _ = env.reset()
        success = True
        
        for i in range(10):
            try:
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                # if the env has a get_fitness_score method, run it
                if hasattr(env, 'get_fitness_score'):
                    fitness_score = env.get_fitness_score()
                    assert fitness_score.shape == (num_envs,)
                assert reward.shape == (num_envs,)
            except Exception as e:

                detailed_error_message = traceback.format_exc()
                success = False
                break
        
        if success:
            return {'success': True, 'diff_functions': diff_functions}
        else:
            return {'success': False, 'diff_functions': diff_functions, 'error_message': detailed_error_message}
        env.close()
    except Exception as e:
        detailed_error_message = traceback.format_exc()
        return {'success': False, 'error_message': detailed_error_message}

def direct_evaluation_run_code(env_id=None, func_dict=None, json_path=None, num_envs=4, sim_backend="gpu", functions_to_overwrite=None):
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(target=lambda q: q.put(run_in_separate_process(
        env_id=env_id,
        func_dict=func_dict,
        json_path=json_path,
        num_envs=num_envs,
        sim_backend=sim_backend,
        functions_to_overwrite=functions_to_overwrite
    )), args=(result_queue,))

    process.start()
    process.join()
    
    # Retrieve the results
    result = result_queue.get()
    return result
    # return run_in_separate_process(env_id=env_id, func_dict=func_dict, json_path=json_path, 
    #                                num_envs=num_envs, sim_backend=sim_backend, functions_to_overwrite=functions_to_overwrite)

if __name__ == '__main__':
    fire.Fire(direct_evaluation_run_code)
