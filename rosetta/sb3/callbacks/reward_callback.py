from stable_baselines3.common.callbacks import EventCallback
import numpy as np
from collections import defaultdict
import inspect

class RewardEvalCallback(EventCallback):
    def __init__(self, env, verbose=1, n_eval_episodes=5,eval_freq=1):
        super().__init__(verbose=verbose)
        self.eval_env = env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
    
    def _on_training_start(self):
        # Call the evaluation method to perform an initial evaluation
        self._on_step()
        
    def _on_step(self) -> bool:
        # Reset the evaluation environment
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            obs = self.eval_env.reset()
            reward_components = defaultdict(float)
            # Initialize variables for evaluation
            num_envs = self.eval_env.num_envs
            completed_episodes = np.zeros(num_envs, dtype=int)
            episode_limit_reached = np.zeros(num_envs, dtype=bool)
            obs = self.eval_env.reset()
        
            # Keep running until all environments have completed `n_episodes`
            while np.any(completed_episodes < self.n_eval_episodes):
                # Get actions from the model
                if "stage" in inspect.getfullargspec(self.model.predict).args:
                    actions, _states = self.model.predict(obs, deterministic=True, stage=self.eval_env.get_attr("cur_stage"))
                else:
                    actions, _states = self.model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, rewards, dones, infos = self.eval_env.step(actions)
                
                # Accumulate rewards and update done episodes
                for i in range(num_envs):
                    if not episode_limit_reached[i]:
                        if "reward_components" in infos[i]:
                            for k,v in infos[i]["reward_components"].items():
                                reward_components[k]+=v
                        if dones[i]:
                            # Episode completed for this environment
                            completed_episodes[i] += 1
                            if completed_episodes[i] >= self.n_eval_episodes:
                                episode_limit_reached[i] = True
            for k,v in reward_components.items():
                self.logger.record(f"reward_components/{k}", v/(num_envs*self.n_eval_episodes))
        
        

        return True

    
