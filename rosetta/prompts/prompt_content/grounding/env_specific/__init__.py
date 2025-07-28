from .ball_and_target import BallAndTarget
from .cube_and_point import CubeAndPoint
from .cube_and_target import CubeAndTarget
from .objects_and_bins import ObjectsAndBins
from .objects_and_drawer import ObjectsAndDrawer
from .sphere_and_bins import SphereAndBins
from .three_cubes import ThreeCubes


ENV_ID_TO_GROUNDING_CLS = {
    # Main short-horizon continuous control
    "BallAndTarget": BallAndTarget,
    "SphereAndBins": SphereAndBins,
    
    # Long-horizon action primitive
    "ObjectsAndBins": ObjectsAndBins,
    "ObjectsAndDrawer": ObjectsAndDrawer,
    "ThreeCubes": ThreeCubes,

    # Baselines + ablations only (short-horizon continuous control)
    "CubeAndPoint": CubeAndPoint,
    "CubeAndTarget": CubeAndTarget
}