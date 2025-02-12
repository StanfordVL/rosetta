from rosetta.prompts.utils import *
from .prompt_env import PromptEnv


class ThreeCubes(PromptEnv):
    def __init__(self):
        self.type = "long_horizon"
        self.description = "Stack 3 cubes on top of each other"
        self.setup_description = ("There are a robot gripper, a red cube, a green cube, and a purple cube on a table. "
                "The 3D coordinates of the projects [x,y,z] are defined from the viewer's perspective: the x-axis represents forward and backward, with positive values being closer and negative values farther away from the viewer; the y-axis denotes horizontal direction, with negative values to the left and positive to the right of the viewer. The z-axis measures height, where z=0 corresponds to the table surface. " 
                "Each cubes are 0.04 by 0.04 by 0.04. Position are measured at each object's center. Expect errors in the measurement.")
        
        self.object_list = ["red_cube", "green_cube", "purple_cube"]
        self.info_keys = ""
        for obj in self.object_list:
            self.info_keys += f"- `{obj}_pos`: 3D coordinate of {obj}\n"

        for obj in self.object_list:
            self.info_keys += f"- `is_{obj}_grasped`: whether {obj} is grasped by the robot\n"
        self.info_keys += f"- `gripper_pos`: 3D coordinate of the robot's gripper\n"
    
    def state_str(self, state, precision = 2):
        red_cube_pos = round_list(state["red_cube_pos"], precision)
        green_cube_pos = round_list(state["green_cube_pos"], precision)
        purple_cube_pos = round_list(state["purple_cube_pos"], precision)
        robot_gripper_pos = round_list(state["gripper_pos"], precision)

        cube_being_hold = None
        if state["is_red_cube_grasped"]:
            cube_being_hold = "red"
        elif state["is_green_cube_grasped"]:
            cube_being_hold = "green"
        elif state["is_purple_cube_grasped"]:
            cube_being_hold = "purple"
        
        return "{\n" + f"Red cube: {red_cube_pos}\nGreen cube: {green_cube_pos}\nPurple cube: {purple_cube_pos}\nRobot gripper: {robot_gripper_pos}\n" + "Robot gripper state: " + \
            (f"hoding a {cube_being_hold} cube" if cube_being_hold else "not holding any cube") + "\n}"

    def action_str(self, action, precision=2):
        target_position = round_list(action["params"], precision)
        if action["action"] == "pick":
            target_position = target_position[:3]
            return f"Pick up at {target_position}"
        elif action["action"] == "place":
            target_position = target_position[:3]
            return f"Place at {target_position}"
        elif action["action"] == "push":
            target_position_1 = target_position[:3]
            target_position_2 = target_position[3:6]
            return f"Push from {target_position_1} to {target_position_2}"
