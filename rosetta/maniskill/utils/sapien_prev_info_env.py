from mani_skill.envs.sapien_env import BaseEnv

class PrevInfoEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super(PrevInfoEnv, self).__init__(*args, **kwargs)
        self.prev_info = None

    def reset(self, *args, **kwargs):
        obs, info= super(PrevInfoEnv, self).reset(*args, **kwargs)
        self.prev_info = info
        return obs, info

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = super(PrevInfoEnv, self).step(*args, **kwargs)
        self.prev_info = info
        return  obs, reward, terminated, truncated, info