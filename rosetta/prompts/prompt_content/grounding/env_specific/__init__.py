from .stack3cube import Stack3Cube
from .sphere2bin_short import Sphere2BinShort
from .push_ball_short import PushBallShort
from .object_to_bin import ObjectToBin
from .put_object_in_drawer import PutObjectInDrawer
from .pull_cube_short import PullCubeShort
from .pick_cube_short import PickCubeShort


ENV_ID_TO_GROUNDING_CLS = {
    "PlaceSphere2BinWide": Sphere2BinShort,
    "PushBall": PushBallShort,
    "Stack3Cube": Stack3Cube,
    "ObjectToBin": ObjectToBin,
    "PutObjectInDrawer": PutObjectInDrawer,
    "Pull1Cube":PullCubeShort,
    "Pick1Cube":PickCubeShort
}