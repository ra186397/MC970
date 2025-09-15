import os

def rename_files_in_folder(folder_path, new_prefix="file_", start_index=1):
    """
    Renames all files in a given folder with a new prefix and sequential numbering.

    Args:
        folder_path (str): The path to the folder containing the files to be renamed.
        new_prefix (str): The desired prefix for the new file names.
        start_index (int): The starting number for the sequential numbering.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    file_list = os.listdir(folder_path)
    current_index = start_index

    for filename in file_list:
        # Construct the full path for the old and new file names
        old_file_path = os.path.join(folder_path, filename)

        # Skip directories, only process files
        if os.path.isfile(old_file_path):
            # Get the file extension
            file_name_without_ext, file_extension = os.path.splitext(filename)

            # Create the new file name with prefix, index, and original extension
            new_file_name = f"{new_prefix}{current_index}{file_extension}"
            new_file_path = os.path.join(folder_path, new_file_name)

            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_file_name}'")
                current_index += 1
            except OSError as e:
                print(f"Error renaming '{filename}': {e}")
        else:
            print(f"Skipping directory: '{filename}'")

if __name__ == "__main__":
    # Specify the folder path where the files are located
    target_folder = "/home/giovani-bianchini/Documents/Disciplinas/MC970/trabalho02/img_tentativa2"  # Replace with your actual folder path

    # Call the function to rename the files
    rename_files_in_folder(target_folder, new_prefix="img", start_index=1)