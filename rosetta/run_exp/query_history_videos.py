from rosetta.run_exp.gen_result_dir import gen_prev_dir_dict
import os
import json
from pathlib import Path
import fire
def get_ordered_history(cur_folder_path: str,result_dirs_path=None):
    # cur_folder_path is a result folder
    # get history including current folder [cur_folder_path,parent1,parent2,...,initial]
    if result_dirs_path is None:
        # get parent folder path
        result_dirs_path = os.path.dirname(cur_folder_path)
    
    folder_dict=gen_prev_dir_dict(result_dirs_path)
    folder_list=[]
    # traverse the prev_id
    def traverse(cur_folder_path):
        with open(os.path.join(cur_folder_path, "exp_config.json"), 'r') as f:
            config = json.load(f)
            prev_id = config.get("prev_uid")
            if prev_id is not None:
                folder_list.append(cur_folder_path)
                traverse(folder_dict[prev_id])
            else:
                folder_list.append(cur_folder_path)
    traverse(cur_folder_path)
    return folder_list

def get_options(cur_folder_path: str,result_dirs_path=None):
    if result_dirs_path is None:
        # get parent folder path
        result_dirs_path = os.path.dirname(cur_folder_path)
    
    with open(os.path.join(cur_folder_path, "exp_config.json"), 'r') as f:
        config = json.load(f)
        uid_feedback = config.get("uid_feedback")
        #round_hash = config.get("round_hash")
    options=[]
    for dir in os.listdir(result_dirs_path):
        abs_dir_path = os.path.join(result_dirs_path, dir)
        with open(os.path.join(abs_dir_path, "exp_config.json"), 'r') as f:
            config = json.load(f)
            if config.get("uid_feedback") == uid_feedback: #and config.get("round_hash") == round_hash:
                options.append(abs_dir_path)
    return options

def single_build_history_and_option_folder(cur_folder_path: str,config_dir_path: str,num_ancestors:int, result_dirs_path=None,save_path='./test'):
    config_dir_name = os.path.basename(config_dir_path)
    # initial doesn't have feedback
    history_folders = get_ordered_history(cur_folder_path,result_dirs_path)

    # If this isn't the right number of ancestors, skip it
    if num_ancestors is not None:
        if num_ancestors != len(history_folders) - 1:
            return 

    os.makedirs(os.path.join(save_path, config_dir_name), exist_ok=True)
    # pprint(history_folders)
    options = get_options(cur_folder_path,result_dirs_path)
    # pprint(options)
    feedback_paths=[]
    video_paths=[]
    for dir in history_folders:
        feedback_path=os.path.join(dir,'grounding_rst.json')
        feedback_paths.append(feedback_path)
        
        # video path should be xxx.mp4, search for the video file
        video_path=None
        for file in os.listdir(dir):
            if file.endswith('.mp4'):
                video_path=os.path.join(dir,file)
                break
        video_paths.append(video_path)

    # pprint(video_paths)
    # pprint(feedback_paths)
    # video-feedback mapping
    video_paths=video_paths[1:]
    feedback_paths=feedback_paths[:-1]
    
    total_paths = len(feedback_paths)
    for idx, pair in enumerate(zip(video_paths,feedback_paths)):
        human_idx = total_paths - idx
        video,feedback=pair
        # copy video and feedback to save_path and rename as video_idx.mp4 and feedback_idx.json
        video_save_path=os.path.join(save_path,config_dir_name,f'video_{human_idx}.mp4')
        with open(feedback, "r") as f:
            feedback_json = json.load(f)
            #feedback_text = feedback_json["original_feedback"]
            feedback_text = feedback_json.get("original_feedback",None)
            if feedback_text is None:
                feedback_text = feedback_json.get("full_feedback",None)
            if feedback_text is None:
                feedback_text = "No feedback"
                print(f"{video_paths} has no feedback")
        feedback_save_path=os.path.join(save_path,config_dir_name,f'your_feedback_on_video_{human_idx}.txt')
        os.system(f'cp {video} {video_save_path}')
        with open(feedback_save_path, "w") as f:
            f.write(feedback_text)
    
    for option in options:
        # copy option to save_path
        folder_path=os.path.join(save_path,config_dir_name,"options-to-choose-from")
        os.makedirs(folder_path,exist_ok=True)
        # get video 
        video_path=None
        for file in os.listdir(option):
            if file.endswith('.mp4'):
                video_path=os.path.join(option,file)
                break
        if video_path is not None:
            video_save_path=os.path.join(folder_path,f'{os.path.basename(video_path)}')
            os.system(f'cp {video_path} {video_save_path}')

def find_relevant_result_folder(config_dir_path,result_dirs_path=None)->list:
    result_folders=[]
    # load config and see if uid_feedback matches
    with open(os.path.join(config_dir_path, "exp_config.json"), 'r') as f:
        config = json.load(f)
        uid_feedback = config.get("uid_feedback")
    for dir in os.listdir(result_dirs_path):
        abs_dir_path = os.path.join(result_dirs_path, dir)
        with open(os.path.join(abs_dir_path, "exp_config.json"), 'r') as f:
            config = json.load(f)
            if config.get("uid_feedback") == uid_feedback:
                result_folders.append(abs_dir_path)
    return result_folders
    
def batch_build_history_and_option_folder(config_dirs_path: str,result_dirs_path:str, num_ancestors:int, save_path='./test'):
    # search at least one relevant result folder for a config_dir
    # config_dir_path_list=list(set(config_dir_path_list))
    # print(type(result_dirs_path))
    # import sys; sys.exit()
    for config_dir_path in Path(config_dirs_path).iterdir():
    # for config_dir_path in config_dir_path_list:
        result_folders=find_relevant_result_folder(config_dir_path,result_dirs_path)
        if len(result_folders)==0:
            print(f"No relevant result folder found for {config_dir_path}")
            continue
        # build history and option folder
        result_folder=result_folders[0]
        single_build_history_and_option_folder(result_folder,config_dir_path,num_ancestors,result_dirs_path,save_path)
    
    print(f"Batch build history and option folder finished, saved to {save_path}")
    
if __name__ == '__main__':
    fire.Fire(batch_build_history_and_option_folder)