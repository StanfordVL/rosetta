from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
class EarlyStoppingEvalCallback(EvalCallback):
    def __init__(self, *args, early_stop_threshold=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_threshold = early_stop_threshold

    def _on_training_start(self):
        # Call the evaluation method to perform an initial evaluation
        self._on_step()
        
    def _on_step(self) -> bool:
        super_return = super()._on_step()
        # After evaluation, check the success rate
        if len(self.evaluations_successes) > 0:
            # Get the last mean success rate
            mean_success_rate = np.mean(self.evaluations_successes[-1])
            if mean_success_rate >= self.early_stop_threshold:
                print(f"Early stopping training because success rate {mean_success_rate} >= {self.early_stop_threshold}")
                self.model.stop_training = True
                return False
        return super_return