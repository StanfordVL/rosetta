from stable_baselines3.common.callbacks import BaseCallback

class SuccessRateCallback(BaseCallback):
    def __init__(self, success_threshold, eval_env, n_eval_episodes=5, verbose=0):
        super(SuccessRateCallback, self).__init__(verbose)
        self.success_threshold = success_threshold
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        # Evaluate the current model
        success_count = 0
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            while not done:
                if "stage" in inspect.getfullargspec(self.model.predict).args:
                    actions, _states = self.model.predict(obs, deterministic=True, stage=self.eval_env.get_attr("cur_stage"))
                else:
                    actions, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                # Assuming that the success is indicated by some flag in info or reward structure
                if done and "is_success" in info[0] and info[0]["is_success"]:
                    success_count += 1
                if done:
                    break
        
        success_rate = success_count / self.n_eval_episodes
        
        if self.verbose > 0:
            print(f"Success Rate: {success_rate:.2f}")
        
        # If success rate exceeds the threshold, stop training
        if success_rate >= self.success_threshold:
            print(f"Stopping training as success rate {success_rate:.2f} exceeds threshold {self.success_threshold:.2f}")
            return False  # This will stop training
        
        return True  # Continue training
