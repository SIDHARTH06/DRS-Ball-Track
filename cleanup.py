import os

# Specify the path to the folder whose contents you want to remove
folder_path = "frames/"

# Loop through all the files and directories in the folder
for filename in os.listdir(folder_path):
    # Construct the full path to the file or directory
    file_path = os.path.join(folder_path, filename)
    
    # Check if the current item is a file
    if os.path.isfile(file_path):
        # If it's a file, delete it using os.remove()
        os.remove(file_path)
    else:
        # If it's a directory, use os.rmdir() to delete it
        os.rmdir(file_path)
folder_path = "masked/"

# Loop through all the files and directories in the folder
for filename in os.listdir(folder_path):
    # Construct the full path to the file or directory
    file_path = os.path.join(folder_path, filename)
    
    # Check if the current item is a file
    if os.path.isfile(file_path):
        # If it's a file, delete it using os.remove()
        os.remove(file_path)
    else:
        # If it's a directory, use os.rmdir() to delete it
        os.rmdir(file_path)

folder_path = "trajectory/"

# Loop through all the files and directories in the folder
for filename in os.listdir(folder_path):
    # Construct the full path to the file or directory
    file_path = os.path.join(folder_path, filename)
    
    # Check if the current item is a file
    if os.path.isfile(file_path):
        # If it's a file, delete it using os.remove()
        os.remove(file_path)
    else:
        # If it's a directory, use os.rmdir() to delete it
        os.rmdir(file_path)
