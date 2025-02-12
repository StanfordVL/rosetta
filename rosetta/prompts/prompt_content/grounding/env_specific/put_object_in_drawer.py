from rosetta.prompts.utils import *
from .task import Task


class PutObjectInDrawer(Task):
    def __init__(self):
        self.description = "Put Apple In Drawer"
        self.setup_description = ("There should be a robot gripper, a red apple, a red soup can, and a drawer on the floor. Objects are roughly 0.05 in high and width. Object positions are measured at the center of the object. "
                "The 3D coordinates of the projects [x,y,z] are defined from the viewer's perspective: the x-axis represents forward and backward, with positive values being closer and negative values farther away from the viewer; the y-axis denotes horizontal direction, with negative values to the left and positive to the right of the viewer. The z-axis measures height, where z=0 corresponds to the floor. " 
                "The bottom drawer is facing the right (+y direction). It's open when the scene starts. The drawer is 0.36 wide, 0.22 deep, and 0.16 high. In other words, x = drawer_pos[0] + 0.18 is the left edge of the drawer, x = drawer_pos[0] - 0.18 is the right edge of the drawer, y = drawer_pos[1] + 0.11 is the drawer front, z = drawer_pos[2] + 0.16 is the top of the drawer, and z = drawer_pos[2] is the bottom of the drawer. "
                "Robot is placed on the floor and cannot reach the inside of the drawer. To put object in the drawer it must be dropped from above.")
        
        self.object_list = ["apple", "soup_can"]
        self.info_keys = ""
        for obj in self.object_list:
            self.info_keys += f"- `{obj}_pos`: 3D coordinate of {obj}\n"

        for obj in self.object_list:
            self.info_keys += f"- `is_{obj}_grasped`: whether {obj} is grasped by the robot\n"
        self.info_keys += f"- `gripper_pos`: 3D coordinate of the robot's gripper\n"
        self.info_keys += f"- `drawer_handle_pos`: 3D coordinate of the bottom drawer handle\n"
        self.info_keys += f"- `drawer_pos`: 3D coordinate of the center bottom of the drawer\n"
        self.info_keys += f"- `drawer_open_offset`: how much the drawer is open, measured as the distance between the drawer's position now and when it is fully closed\n"

    def state_str(self, state, precision = 2):
        apple_pos = round_list(state["apple_pos"], precision)
        soup_can_pos = round_list(state["soup_can_pos"], precision)
        gripper_pos = round_list(state["gripper_pos"], precision)
        drawer_handle_pos = round_list(state["drawer_handle_pos"], precision)
        drawer_center_pos = round_list(state["drawer_pos"], precision)
        drawer_open_offset = round(state["drawer_open_offset"], precision)

        apple_in_drawer = (apple_pos[2] > 0.19) and (drawer_center_pos[0] - 0.18 < apple_pos[0] < drawer_center_pos[0] + 0.18) and (drawer_center_pos[1] - 0.18 < apple_pos[1] < drawer_center_pos[1] + 0.18)
        soup_can_in_drawer = (soup_can_pos[2] > 0.19) and (drawer_center_pos[0] - 0.18 < soup_can_pos[0] < drawer_center_pos[0] + 0.18) and (drawer_center_pos[1] - 0.18 < soup_can_pos[1] < drawer_center_pos[1] + 0.18)

        holding = False
        if state["is_apple_grasped"]:
            holding = "apple"
        elif state["is_soup_can_grasped"]:
            holding = "soup can"

        return "{\n" + f"Apple position: {apple_pos}\nSoup Can Position: {soup_can_pos}\nDrawer Handle position: {drawer_handle_pos}\nDrawer position: {drawer_center_pos}\nDrawer is opened {drawer_open_offset} meter\n" + f"Robot gripper position: {gripper_pos}\nRobot gripper state: " + \
            (f"hoding the {holding}" if holding else "not holding anything") + \
            ("\nApple is in the drawer" if apple_in_drawer  else "\nApple is not in the drawer") + ("\nSoup Can is in the drawer" if soup_can_in_drawer  else "\nSoup Can is not in the drawer") + \
            "\n}"

    def action_str(self, action, precision=2):
        target_position = round_list(action["params"], precision)
        if action["action"] == "pick":
            return f"Pick up at {target_position[0:3]}"
        elif action["action"] == "place":
            return f"Place at {target_position[0:3]}"
        elif action["action"] == "push":
            return f"Push from {target_position[0:3]} to {target_position[3:6]}"
