from .ThreeCubes import ThreeCubes
from .sphere_and_bins import SphereAndBins
from .ball_and_target import BallAndTarget
from .objects_and_bins import ObjectsAndBins
from .objects_and_drawer import ObjectsAndDrawer
from .cube_and_target import CubeAndTarget
from .cube_and_point import CubeAndPoint


ENV_ID_TO_GROUNDING_CLS = {
    "SphereAndBins": SphereAndBins,
    "BallAndTarget": BallAndTarget,
    "ThreeCubes": ThreeCubes,
    "ObjectsAndBins": ObjectsAndBins,
    "ObjectsAndDrawer": ObjectsAndDrawer,
    "CubeAndTarget":CubeAndTarget,
    "CubeAndPoint":CubeAndPoint
}