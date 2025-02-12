from feedback_to_reward_prompts.prompt_message import *
from feedback_to_reward_prompts.prompt_designs import *
from feedback_to_reward_prompts.utils import *


DESIGN_NAME_TO_FUNC = {

    # Short-horizon continuous control
    "rosetta_sh": (rosetta_sh, "rosetta_sh/rosetta_sh"),
    "rosetta_sh_refined": (rosetta_sh, "rosetta_sh/rosetta_sh_refined"),
    "rosetta_sh_nohistory": (rosetta_sh_nohistory, "rosetta_sh/rosetta_sh_nohistory"),

    # Long-horizon action primitives
    "rosetta_lh": (rosetta_lh, "rosetta_lh/rosetta_lh"),

    # Baselines
    "eureka": (eureka, "baselines/eureka"),
    "t2r": (text2reward, "baselines/text2reward"),
    
    # Ablations
    "no_grounding": (no_grounding, "contcontrol/no_grounding"),
    "no_follow_up": (no_follow_up, "contcontrol/rosetta"),
    "no_staging": (no_staging, "ablations/no_staging"),
}
