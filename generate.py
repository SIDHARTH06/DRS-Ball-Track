import cv2
import os
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
def gen(path):
# Path to the directory containing the frames
    frames_dir = "trajectory"

# Get the names of all the frames in the directory
    frame_names = sorted_alphanumeric(os.listdir(frames_dir))

# Define the codec to be used for the output video
    fourcc = cv2.VideoWriter_fourcc(*"h264")

# Define the frame size and frames per second for the output video
    frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    frame_size =  frame.shape[:2][::-1]
    fps = 30

# Create a VideoWriter object to write the output video
    output_video = cv2.VideoWriter("static/output.mp4", fourcc, fps, frame_size)

# Loop through all the frames, read them, and write them to the output video
    for frame_name in frame_names:
    # Construct the full path to the frame
        frame_path = os.path.join(frames_dir, frame_name)

    # Read the frame
        frame = cv2.imread(frame_path)

    # Resize the frame to the desired size
        frame = cv2.resize(frame, frame_size)

    # Write the frame to the output video
        output_video.write(frame)

# Release the VideoWriter object
    output_video.release()
