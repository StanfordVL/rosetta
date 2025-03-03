import json
import os
from pathlib import Path
import fire

from typing import Tuple
import re
from rosetta.run_exp.utils import generate_hash_uid

# Key mapping dictionary
KEY_MAP = {
    "What's your name? Make sure to enter your name the same way every single time.": "annotator_name",
    "Which video are you giving feedback for? Copy-paste the .mp4 file. So if the file is named \"video_name.mp4\", put \"video_name\" here.": "video_name",
    "Give feedback. Again, be natural! Speak the way you'd want to speak to your personal robot assistant. Even if the task isn't inherently interesting, feel free to be creative - ask to change the order, the location, and more.": "feedback",
    "Describe the video": "description"
}

def parse_videoname(filename: str) -> Tuple[str, str, str]:
    """
    Parse a filename with format {annotator_id}-{env_id}-{uid} into its components.

    Args:
        filename (str): The filename to parse

    Returns:
        Tuple[str, str, str]: (annotator_id, env_id, uid)

    Examples:
        >>> parse_videoname("isabella-PlaceSphere2BinWide-b44eadad34")
        ('isabella', 'PlaceSphere2BinWide', 'b44eadad34')
        >>> parse_videoname("fatima-test-Pushball-42286098d9.mp4")
        ('fatima-test', 'Pushball', '42286098d9')

    Notes:
        - env_id: must contain at least one uppercase letter
        - uid: must be alphanumeric (at the end, after last hyphen)
        - annotator_id: may contain letters, numbers, underscores, and hyphens
    """
    # Remove directory path if present and file extension
    base_name = filename.split('/')[-1].split('\\')[-1].split('.')[0]

    # Find uid at the end (alphanumeric string)
    # Split by last hyphen to get uid
    parts = base_name.rsplit('-', 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot find valid UID in filename: {filename}")
    uid = parts[1]
    if not uid.isalnum():
        raise ValueError(f"Invalid UID format (must be alphanumeric): {uid}")

    remaining = parts[0]
    parts = remaining.split('-')
    env_id = None
    annotator_parts = []

    for part in parts:
        if any(c.isupper() for c in part):
            env_id = part
            break
        annotator_parts.append(part)

    if env_id is None:
        raise ValueError(f"Cannot find valid env_id in filename: {filename}")
    annotator_id = '-'.join(annotator_parts)

    return annotator_id, env_id, uid


def parse_raw_dict(raw_dict, key_map=KEY_MAP):
    """
    Parse a raw dictionary into a new dictionary with mapped keys

    Args:
        raw_dict (dict): The raw dictionary to parse
        key_map (dict): A dictionary mapping old keys to new keys

    Returns:
        dict: A new dictionary with mapped keys
    """
    new_dict = {}
    for old_key, new_key in key_map.items():
        new_dict[new_key] = raw_dict.get(old_key, "")
    annotator_id, env_id, prev_uid= parse_videoname(new_dict["video_name"])
    new_dict.update({
        "annotator_id": annotator_id,
        "env_id": env_id,
        "prev_uid": prev_uid
    })

    return new_dict

def gen_config_dirs(src_path, save_dir='./test_input', key_map=KEY_MAP):
    """
    Process feedback data from a JSONL file and create organized folder structure

    Args:
        src_path (str): Path to the source JSONL file
        save_dir (str): Path where the folder structure will be created

    Creates:
        --save_dir
            --{annotator_id}_{env_id}_{prev_uid}_{uid_feedback}
                --exp_config.json
            --...
    Returns:
        list: List of newly created folder names
    """
    new_folders = []  # Track newly created folders

    with open(src_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            try:
                raw_dict = json.loads(line.strip())
                parsed_dict = parse_raw_dict(raw_dict, key_map=KEY_MAP)

                if len(parsed_dict["annotator_id"])==0:
                    parsed_dict["annotator_id"] = parsed_dict["annotator_name"].split(" ")[0].lower()
                uid_feedback = generate_hash_uid(str(parsed_dict["annotator_id"])+
                                          str(parsed_dict["env_id"])+
                                          str(parsed_dict["prev_uid"])+
                                          str(parsed_dict["feedback"]))
                parsed_dict.update({
                    "uid_feedback": uid_feedback
                })
                folder_name = f"{parsed_dict['annotator_id']}-{parsed_dict['env_id']}-{parsed_dict['prev_uid']}-{uid_feedback}"
                folder_path = os.path.join(save_dir, folder_name)
                config_path = os.path.join(folder_path, "exp_config.json")


                if os.path.exists(folder_path) and os.path.exists(config_path):
                    continue
                os.makedirs(folder_path, exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(parsed_dict, f, indent=4)
                new_folders.append(folder_path)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_number}:")
                print(f"Line content: {line.strip()}")
                print(f"Error details: {str(e)}")
                continue

            except KeyError as e:
                print(f"Missing key in dictionary at line {line_number}:")
                print(f"Missing key: {str(e)}")
                print(f"Available keys: {list(parsed_dict.keys() if 'parsed_dict' in locals() else raw_dict.keys())}")
                continue

            except OSError as e:
                print(f"File system error at line {line_number}:")
                print(f"Error creating directory or writing file: {str(e)}")
                continue

            except Exception as e:
                print(f"Unexpected error at line {line_number}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {str(e)}")
                print("Stack trace:")
                import traceback
                traceback.print_exc()
                continue

    return new_folders