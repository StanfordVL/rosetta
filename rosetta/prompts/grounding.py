import cv2
import glob
import json
import os
from pathlib import Path
from IPython.display import Video, display

from rosetta.prompts.prompt_content.grounding.env_specific import (
    ENV_ID_TO_GROUNDING_CLS,
)
from rosetta.prompts.utils import *

### Step 1: Convert demonstration to natural language description
def state_description(data, env, act_space, client, scale=1):
    """ Generate natural language description of a state """

    # Convert image to base64
    img_b64_str, img_type = image_path_to_base64(data["image_path"], scale)

    # prepare messages
    state_sys_msg = PromptMessage(
        role="system", content=get_prompt_content(f"grounding/state_desc_system")
    )
    state_usr_msg = PromptMessage(
        role="user",
        content=get_prompt_content(f"grounding/{act_space}/state_desc_user"),
    )
    state_usr_msg.fill_dynamic_fields(
        {
            "env_setup_description": env.setup_description,
            "env_state_str": env.state_str(data),
        }
    )

    # Add image to the user message
    state_usr_msg.content = [
        {"type": "text", "text": state_usr_msg.content},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
        },
    ]
    hist = [state_sys_msg, state_usr_msg]

    # Query GPT-4o for state description
    params = {
        "temperature": 0.1,
        "max_completion_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 1.0,
    }
    return query_until_complete(client, hist, "gpt-4o", params).content


def action_description(action_state, env, act_space, client):
    """ Generate natural language description of an action """
    # Get the action description
    start_state_coordinate = env.state_str(action_state["start_state"])
    start_state_language_description = action_state["start_state"]["description"]
    end_state_coordinate = env.state_str(action_state["end_state"])
    end_state_language_description = action_state["end_state"]["description"]

    # Prepare prompt
    system_actdesc_msg = PromptMessage(
        role="system", content=get_prompt_content(f"grounding/act_desc_system")
    )
    user_actdesc_msg = PromptMessage(
        role="user", content=get_prompt_content(f"grounding/{act_space}/act_desc_user")
    )
    dynamic_fields = {
        "env_setup_description": env.setup_description,
        "start_state_coordinate": start_state_coordinate,
        "start_state_language_description": start_state_language_description,
        "end_state_coordinate": end_state_coordinate,
        "end_state_language_description": end_state_language_description,
        "action_description": action_description,
    }
    # Add action description to the user message, action is only interpretable in the actprim case
    if act_space == "actprim":
        dynamic_fields["action_description"] = action_description
    user_actdesc_msg.fill_dynamic_fields(dynamic_fields)
    hist = [system_actdesc_msg, user_actdesc_msg]

    # Query GPT-4o for action description
    params = {
        "temperature": 0.1,
        "max_completion_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 1.0,
    }
    return query_until_complete(client, hist, "gpt-4o", params).content


def demo_to_language_description(demo_dir, env_id, act_space, client, verbose=True):
    """
    Convert a demonstration to natural language using GPT

    Args:
        demo_dir (str): The directory containing the demonstration with subdirectories "video" and "trajectory.json". The
                        video directory contains the video files for each step of the demonstration, and the trajectory.json
                        file contains the state information for each step.
    """

    # Load the environment
    env = ENV_ID_TO_GROUNDING_CLS[env_id]()
    # Load the demo data
    traj_dir = os.path.join(demo_dir, "trajectory.json")
    with open(traj_dir) as f:
        traj = json.load(f)

    # Actprim split the trajectory by high level actions, contcontrol is split into states uniformly
    if act_space == "actprim":
        # Load the video data
        video_dir = os.path.join(demo_dir, "video")
        video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
        video_paths.sort(
            key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1])
        )
        num_steps = len(video_paths)
        assert (
            len(traj) == num_steps * 2 + 1
        ), "Mismatch between number of video files and number of steps, {} != {}".format(
            len(traj), num_steps
        )
        if verbose:
            print("Loaded demo with {} steps".format(num_steps))

        # Create folder to dump frames
        image_folder = os.path.join(demo_dir, "frames")
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # Generate state and action descriptions
        for i in range(num_steps):
            start_frame = get_first_frame(video_paths[i])
            end_frame = get_last_frame(video_paths[i])

            start_frame_path = os.path.join(image_folder, f"action_{i}_start_frame.png")
            end_frame_path = os.path.join(image_folder, f"action_{i}_end_frame.png")

            # save images to the path
            cv2.imwrite(start_frame_path, start_frame)
            cv2.imwrite(end_frame_path, end_frame)

            # Add image path to the state
            traj[i * 2]["image_path"] = start_frame_path

            # Generate state description for each state
            traj[i * 2]["description"] = state_description(
                traj[i * 2], act_space=act_space, client=client, env=env
            )
            if verbose:
                print(f"State {i} description: {traj[i*2]['description']}")

        # Handle the last state
        traj[-1]["image_path"] = end_frame_path
        traj[-1]["description"] = state_description(
            traj[-1], act_space=act_space, client=client, env=env
        )
        if verbose:
            print(f"State {num_steps} description: {traj[-1]['description']}")

        # Store state descriptions
        with open(
            os.path.join(demo_dir, "traj_with_state_descriptions.json"), "w"
        ) as f:
            json.dump(traj, f, indent=4)

        # Generate Action Descriptions
        action_infos = []
        for i in range(num_steps):
            action_info = {
                "start_state": traj[i * 2],
                "action": traj[i * 2 + 1],
                "end_state": traj[i * 2 + 2],
                "video_path": video_paths[i],
            }
            action_infos.append(action_info)

        # Generate action descriptions
        for i in range(num_steps):
            res = action_description(
                action_infos[i], act_space=act_space, client=client, env=env
            )
            traj[i * 2 + 1]["description"] = res

            if verbose:
                print(f"Step {i} action description: {res}")

        # Save Trajectory with State and Action description
        with open(
            os.path.join(demo_dir, "traj_with_state_and_action_descriptions.json"), "w"
        ) as f:
            json.dump(traj, f, indent=4)

        # keep only the language description
        description = {}
        state_count = 0
        is_state = True
        for t in traj:
            if is_state:
                description[f"state_{state_count}"] = t["description"]
                is_state = False
            else:
                description[f"action_{state_count}"] = t["description"]
                state_count += 1
                is_state = True
    elif act_space == "contcontrol":
        # Load the frames
        image_folder = os.path.join(demo_dir, "frames")
        image_paths = glob.glob(os.path.join(image_folder, "*.png"))
        image_paths.sort(
            key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1])
        )
        num_frames = len(image_paths)
        assert (
            len(traj) == num_frames
        ), f"Mismatch between number of steps and number of frame files {len(traj)} != {num_frames}"
        if verbose:
            print("Loaded demo with {} frames".format(num_frames))

        # Generate state descriptions
        for i in range(num_frames):
            traj[i]["image_path"] = image_paths[i]
            traj[i]["description"] = state_description(
                traj[i], act_space=act_space, client=client, env=env
            )
            if verbose:
                print(f"State {i} description: {traj[i]['description']}")
        with open(
            os.path.join(demo_dir, "traj_with_state_descriptions.json"), "w"
        ) as f:
            json.dump(traj, f, indent=4)

        # Generate action descriptions and store all descriptions
        description = {}
        for i in range(num_frames - 1):
            action_info = {"start_state": traj[i], "end_state": traj[i + 1]}
            res = action_description(
                action_info, act_space=act_space, client=client, env=env
            )

            description[f"state_{i}"] = traj[i]["description"]
            description[f"action_{i}"] = res

            if verbose:
                print(f"Step {i} action description: {res}")

        # Handle the last state
        description[f"state_{num_frames-1}"] = traj[-1]["description"]
    else:
        raise ValueError("Invalid action space")

    with open(os.path.join(demo_dir, "description.json"), "w") as f:
        json.dump(description, f, indent=4)

    return description


### Step 2: Ground the preference
def prompt_ground_preference(
    demo_description, original_preference, task_description, client, temperature
):
    # Generate the demo language description
    demo_string = ""
    for description_key, description_value in demo_description.items():
        demo_string += f"{description_key}: {description_value}\n"

    # prepare prompt
    ground_system_msg = PromptMessage(
        role="system", content=get_prompt_content("grounding/ground_preference_system")
    )
    ground_user_msg = PromptMessage(
        role="user", content=get_prompt_content("grounding/ground_preference_user")
    )
    ground_user_msg.fill_dynamic_fields(
        {
            "task_description": task_description,
            "video_description": demo_string,
            "original_preference": original_preference,
        }
    )
    hist = [ground_system_msg, ground_user_msg]

    # Query GPT-4o for grounded preference
    params = {
        "temperature": temperature,
        "max_completion_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 1.0,
        "response_format": "json_object",
    }
    content = query_until_complete(client, hist, "gpt-4o", params).content
    json_obj = json.loads(content)
    return json_obj


### Step 3: Generate a new task description based on the grounded preference
def prompt_new_description(
    demo_description, task_description, grounded_preference, client, temperature
):
    # prepare prompt
    new_desc_system_msg = PromptMessage(
        role="system", content=get_prompt_content("grounding/new_description_system")
    )
    new_desc_user_msg = PromptMessage(
        role="user", content=get_prompt_content("grounding/new_description_user")
    )
    new_desc_user_msg.fill_dynamic_fields(
        {
            "task_description": task_description,
            "demo_description": demo_description,
            "grounded_preference": grounded_preference,
        }
    )
    hist = [new_desc_system_msg, new_desc_user_msg]

    # Query GPT-4o for new task description
    params = {
        "temperature": temperature,
        "max_completion_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 1.0,
        "response_format": "json_object",
    }
    content = query_until_complete(client, hist, "gpt-4o", params).content
    json_obj = json.loads(content)
    return json_obj["task_description"]


def ground_preference(
    demo_dir,
    preference_text,
    env_id,
    act_space,
    client,
    task_description,
    model="gpt-4o",
    temperature=0.1,
    **kwargs,
):
    """
    Takes in a demonstration and a user prefernce and returns a grounded preference and a new task description
    """

    # Load the environment's prompt object
    env = ENV_ID_TO_GROUNDING_CLS[env_id]()
    if task_description is None:
        task_description = env.description

    # Generate language description of the demo
    if ("description_exists" not in kwargs) or (not kwargs["description_exists"]):
        preference_file = os.path.join(demo_dir, "preference.txt")
        try:
            with open(preference_file, "w") as f:
                f.write(preference_text)
        # TODO remove if possible.
        except FileNotFoundError:
            legacy_demo_dir = Path(demo_dir).parent / "episode_0"
            legacy_preference_file = legacy_demo_dir / "preference.txt"
            with open(legacy_preference_file, "w") as f:
                f.write(preference_text)

        demo_description = demo_to_language_description(
            demo_dir=demo_dir,
            act_space=act_space,
            client=client,
            env_id=env_id,
            verbose=True,
        )
    else:
        descrip_fn = Path(demo_dir) / "description.json"
        with open(descrip_fn, "r") as f:
            demo_description = json.load(f)

    # Generate grounded preference and Summary
    output = prompt_ground_preference(
        demo_description,
        preference_text,
        task_description,
        client=client,
        temperature=temperature,
    )

    # Get new task description based on this preference, for use in the *next* round
    new_task_description = prompt_new_description(
        demo_description=demo_description,
        task_description=task_description,
        grounded_preference=output["preference"],
        client=client,
        temperature=temperature,
    )

    # extract only the relevant information
    output["grounded_preference"] = output.pop("preference")
    output["task_description"] = task_description
    output["next_description"] = new_task_description
    output["original_preference"] = preference_text
    return output
