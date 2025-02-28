import base64 
import cv2
import glob
from IPython.display import Video, display
import json
import os
from PIL import Image
import subprocess


def round_list(lst, precision):
    return [round(x, precision) for x in lst]


def image_path_to_base64(image_path, scale = 1): 
    # Load and convert the image to base64 string
    
    if scale != 1:
        img = Image.open(image_path)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        scaled_image_path = f"tmp_resized_{scale}.png"
        img.save(scaled_image_path)
        with open(scaled_image_path, "rb") as image_file:
            image_data = image_file.read()
            img_b64_str = base64.b64encode(image_data).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            img_b64_str = base64.b64encode(image_data).decode("utf-8")
    
    # Determine the image MIME type
    img_type = "image/jpeg" if image_path.endswith(".jpg") else "image/png"
    
    # Display the assistant's response
    return img_b64_str, img_type


def show_traj(dir):
    video_dir = os.path.join(dir, "video")
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    video_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    traj_dir = os.path.join(dir, "trajectory.json")
    with open(traj_dir) as f:
        traj = json.load(f)
    
    for i in range(len(video_paths)):
        print(f"STEP {i}:")
        print(traj[i*2])
        print(traj[i*2+1])
        display(Video(video_paths[i]))
    
    print("FINAL STATE:")
    print(traj[-1])


def stitch_mp4_files(input_folder, output_file):
    # Get a list of all MP4 files in the input folder
    dirs = os.listdir(input_folder)

    # sorted by file name as a number
    dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))

    mp4_files = [file for file in dirs if file.endswith(".mp4")]

    # Create a list of input file paths
    input_files = [os.path.join(input_folder, file) for file in mp4_files]
    
    # Create the ffmpeg command to stitch the files together
    ffmpeg_command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", output_file]

    # Create a temporary input file containing the list of input files
    with open("input.txt", "w") as f:
        for file in input_files:
            f.write(f"file '{file}'\n")

    # Run the ffmpeg command
    subprocess.run(ffmpeg_command)

    # Remove the temporary input file
    os.remove("input.txt")


def get_first_frame(video_file):
    # Open the video file
    video = cv2.VideoCapture(video_file)
    
    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return None
    
    # Read the first frame
    ret, frame = video.read()
    
    # Check if the frame was successfully read
    if not ret:
        print("Error reading video frame")
        return None
    
    # Release the video file
    video.release()
    
    return frame


def get_last_frame(video_file):
    # Open the video file
    video = cv2.VideoCapture(video_file)
    
    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return None

    
    # Read the video frame by frame
    last_frame = None
    while True:
        ret, frame = video.read()
        
        # If there are no more frames, break the loop
        if not ret:
            break
        else:
            last_frame = frame
        
    # Release the video file
    video.release()
    
    return last_frame