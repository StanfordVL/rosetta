# Short horizon continuous control
from .rosetta_sh.rosetta_sh import rosetta_sh
from .rosetta_sh.rosetta_sh_nohistory import rosetta_sh_nohistory

# Long horizon action primitive
from .rosetta_lh.rosetta_lh import rosetta_lh

# Baselines
from .baselines.eureka import eureka
from .baselines.text2reward import text2reward
from .baselines.lmpc import lmpc

# Ablations
from .ablations.no_follow_up import no_follow_up
from .ablations.no_grounding import  no_grounding
from .ablations.no_staging import no_staging

from .design_names_to_funcs import DESIGN_NAME_TO_FUNC 