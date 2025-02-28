from stable_baselines3.common.callbacks import EvalCallback

class EvalCallbackAtStart(EvalCallback):
    def _on_training_start(self):
        # Call the evaluation method to perform an initial evaluation
        self._on_step()