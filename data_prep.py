import os
import shutil
from random import sample

LIB_DIRECTORY = "lib_unzippped"
TRAIN_DIRECTORY = 'train_data'
TEST_DIRECTORY = "test_data"
FILE_EXTENSIONS = ['.py', '.cpp', '.c', '.h']


def copy_files(dst_dir, file_list):
    """
    Copies the list of files to the destination directory.

    Args:
        dst_dir (str): The destination directory to copy files to.
        file_list (list): A list of file paths to be copied.

    Returns:
        list: A list of copied file names.
    """
    copied_files = []
    for file in file_list:
        # Get the destination path for the file
        destination_file_path = os.path.join(dst_dir, os.path.basename(file))
        # Copy the file to the destination directory
        shutil.copy2(file, destination_file_path)
        # Add the basename of the copied file to the list
        copied_files.append(os.path.basename(file))
    return copied_files


def move_percentage_files(src_dir, dst_dir, file_list, percentage):
    """
    Moves a percentage of files from the source directory to the destination directory.

    Args:
        src_dir (str): The source directory to move files from.
        dst_dir (str): The destination directory to move files to.
        file_list (list): A list of file names to be moved.
        percentage (float): The percentage of files to move (between 0 and 1).

    Returns:
        None
    """
    # Calculate how many files to move based on the percentage
    num_to_move = max(1, int(len(file_list) * percentage))
    if file_list:  # Proceed only if the file list is not empty
        # Randomly select the files to move
        files_to_move = sample(file_list, num_to_move)
        for file in files_to_move:
            # Construct full source and destination file paths
            source_file_path = os.path.join(src_dir, file)
            destination_file_path = os.path.join(dst_dir, file)
            try:
                # Move the file from source to destination
                shutil.move(source_file_path, destination_file_path)
            except Exception:
                # In case of error (e.g., file doesn't exist), pass without interruption
                pass


if __name__ == '__main__':
    # Create directories for training and testing data if they don't exist
    if not os.path.exists(TRAIN_DIRECTORY):
        os.makedirs(TRAIN_DIRECTORY)

    if not os.path.exists(TEST_DIRECTORY):
        os.makedirs(TEST_DIRECTORY)

    # Dictionary to hold file paths categorized by their extensions
    file_paths = {ext: [] for ext in FILE_EXTENSIONS}

    # Walk through the source directory and collect file paths based on the extensions
    for root, dirs, files in os.walk(LIB_DIRECTORY):
        for file in files:
            for ext in FILE_EXTENSIONS:
                if file.endswith(ext):
                    file_paths[ext].append(os.path.join(root, file))

    # Dictionary to store the files copied to the training directory
    copied_files = {ext: [] for ext in FILE_EXTENSIONS}
    for ext, file_list in file_paths.items():
        # Copy the files to the training directory
        copied_files[ext] = copy_files(TRAIN_DIRECTORY, file_list)

    print('---- Training data prepared ----')
    print('---- Moving 20 % to test data ----')

    # Move 20% of the copied files from the training directory to the testing directory
    for ext, file_list in copied_files.items():
        move_percentage_files(TRAIN_DIRECTORY, TEST_DIRECTORY, file_list, 0.2)

    # Output the number of files in the train and test directories
    print('--------------------------------------')
    print('Train file num: ', len(os.listdir(TRAIN_DIRECTORY)))
    print('.py:', len([f for f in os.listdir(TRAIN_DIRECTORY) if f.endswith('.py')]))
    print('.cpp:', len([f for f in os.listdir(TRAIN_DIRECTORY) if f.endswith(('.cpp', '.c', '.h'))]))

    print('Test file num: ', len(os.listdir(TEST_DIRECTORY)))
    print('.py:', len([f for f in os.listdir(TEST_DIRECTORY) if f.endswith('.py')]))
    print('.cpp:', len([f for f in os.listdir(TEST_DIRECTORY) if f.endswith(('.cpp', '.c', '.h'))]))
