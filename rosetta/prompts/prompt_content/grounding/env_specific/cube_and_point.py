from rosetta.prompts.utils import *
from .prompt_env import PromptEnv


class CubeAndPoint(PromptEnv):
    def __init__(self):
        self.description = "Pick a cube to goal position"
        self.setup_description = """There should be a robot gripper and a cube on the table.
The 3D coordinates of the projects [x,y,z] are defined from the viewer’s perspective: the x-axis represents forward and backward, with positive values being closer and negative values farther away from the viewer; the y-axis denotes horizontal direction, with negative values to the left and positive to the right of the viewer. This is the opposite of typical - check your work!

The z-axis measures height, where z=0 corresponds to the table surface. Positions are measured at each object's center. Expect errors in the measurement. 
"""
  
    def state_str(self, state, precision = 2):
        ball_pos = round_list(state["cube_pos"], precision)
        goal_pos = round_list(state["goal_pos"], precision)
        robot_gripper_pos = round_list(state["gripper_pos"], precision)

        return "{\n" + f"Ball: {ball_pos}\nGoal: {goal_pos}\nRobot gripper: {robot_gripper_pos}\n" + "Robot gripper state: " + \
            (f"is not holding anything") + "\n}"

    def action_str(self, action, precision=2):
        pass
