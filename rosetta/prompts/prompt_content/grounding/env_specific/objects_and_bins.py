from rosetta.prompts.utils import *
from .prompt_env import PromptEnv


class ObjectsAndBins(PromptEnv):
    def __init__(self):
        self.description = "Put some objects in bins."
        self.setup_description = ("There should be a robot gripper, an apple, an orange, a baseball, a tennis ball, and two bins on the table. One bin is light blue and one is white."
                "The 3D coordinates of the projects [x,y,z] are defined from the viewer's perspective: the x-axis represents forward and backward, with positive values being closer and negative values farther away from the viewer; the y-axis denotes horizontal direction, with negative values to the left and positive to the right of the viewer. The z-axis measures height, where z=0 corresponds to the table surface. " 
                "Each objects are about 0.05 in diameter. Each bins are 0.20 by 0.20 by 0.02 in size. Position are measured at each object's center. Expect errors in the measurement.")
        
        self.object_list = ["apple", "orange", "baseball", "tennis_ball", "blue_bin", "white_bin"]

        self.info_keys = ""
        for obj in self.object_list:
            self.info_keys += f"- `{obj}_pos`: 3D coordinate of {obj}\n"

        for obj in self.object_list:
            if "bin" in obj:
                continue
            else:
                self.info_keys += f"- `is_{obj}_grasped`: whether {obj} is grasped by the robot\n"
        self.info_keys += f"- `gripper_pos`: 3D coordinate of the robot's gripper\n"
    
    def state_str(self, state, precision = 2):
        string = "{\n"

        for obj in self.object_list:
            obj_pos = round_list(state[f"{obj}_pos"], precision)
            string += f"{obj}: {obj_pos}\n"
            
        is_holding = None

        for obj in self.object_list:
            if "bin" in obj:
                continue
            if state[f"is_{obj}_grasped"]:
                is_holding = obj
                break
            
        string += f"Robot gripper: {round_list(state['gripper_pos'], precision)}\n"
        string += f"Robot gripper state: {f'holding the {is_holding}' if is_holding else 'not holding any objects.'}\n"
        string +=  "\n}"

        return string

    def action_str(self, action, precision=2):
        target_position = round_list(action["params"], precision)
        if action["action"] == "pick":
            return f"Pick up at {target_position[0:3]}"
        elif action["action"] == "place":
            return f"Place at {target_position[0:3]}"
        elif action["action"] == "push":
            return f"Push from {target_position[0:3]} to {target_position[3:6]}"
