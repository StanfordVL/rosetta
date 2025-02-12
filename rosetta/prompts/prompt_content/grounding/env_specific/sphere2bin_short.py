from feedback_to_reward_prompts.utils import *
from .task import Task


class Sphere2BinShort(Task):

    bin_base_half_len = 0.08
    bin_height = 0.015
    radius = 0.02

    def __init__(self):
        self.description = "Put a sphere into the bin"
        self.setup_description = """There should be a robot gripper, a ball, and a target on the table. 
The 3D coordinates of the projects [x,y,z] are defined from the viewer’s perspective: the x-axis represents forward and backward, with positive values being closer and negative values farther away from the viewer; the y-axis denotes horizontal direction, with negative values to the left and positive to the right of the viewer. This is the opposite of typical - check your work!

The z-axis measures height, where z=0 corresponds to the table surface. Positions are measured at each object's center. Expect errors in the measurement. 
"""

    def is_sphere_in_bin(self, sphere_pos, bin_pos):
        return (abs(sphere_pos[0] - bin_pos[0]) < self.bin_base_half_len) and \
            (abs(sphere_pos[1] - bin_pos[1]) < self.bin_base_half_len) and \
            ((sphere_pos[2] - bin_pos[2]) < self.bin_height + self.radius)
    
    def state_str(self, state, precision = 2):
        sphere_pos = round_list(state["sphere_pos"], precision)
        bin2_pos = round_list(state["bin_2_pos"], precision)
        bin1_pos = round_list(state["bin_1_pos"], precision)
        robot_gripper_pos = round_list(state["gripper_pos"], precision)

        cube_being_hold = None
        if state["is_sphere_grasped"]:
            cube_being_hold = "sphere"

        shpere_in_bin1 = self.is_sphere_in_bin(sphere_pos, bin1_pos)
        shpere_in_bin2 = self.is_sphere_in_bin(sphere_pos, bin2_pos)
          
        state_string = "{\n" + f"Sphere: {sphere_pos}\nBin 1: {bin1_pos}\nBin 2: {bin2_pos}\nRobot gripper: {robot_gripper_pos}\n" + "Robot gripper state: " + \
            (f"hoding the sphere" if cube_being_hold else "not holding anything, ") + \
            (f"\nSphere is inside bin 1: {shpere_in_bin1}, ") + \
            (f"\nSphere is inside bin 2: {shpere_in_bin2}, ") + "\n}"
    
        return state_string

    def action_str(self, action, precision=2):
        pass
