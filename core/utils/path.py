import os


def get_current_path(file_name: str) -> str:
    return os.path.dirname(os.path.realpath(file_name))


def to_absolute_path(rel_path: str) -> str:
    return os.path.abspath(rel_path)


def get_cwd() -> str:
    return os.getcwd()


def get_num_folders_with_prefix(prefix: str, parent_folder: str) -> int:
    num_folders = 0
    for folder in os.listdir(parent_folder):
        if folder.startswith(prefix):
            num_folders += 1
    return num_folders


def build_child_path_with_prefix(prefix: str, parent_folder: str) -> str:
    num_folders = get_num_folders_with_prefix(prefix, parent_folder)

    return f"{prefix}_{num_folders:03d}"


def build_full_path_with_prefix(prefix: str, parent_folder: str) -> str:
    num_folders = get_num_folders_with_prefix(prefix, parent_folder)

    return os.path.join(parent_folder, f"{prefix}_{num_folders:03d}")


def get_folder_from_path(file_path: str) -> str:
    return os.path.dirname(file_path)


def get_files_in_folder(folder: str) -> list:
    return os.listdir(folder)


def get_files_with_extension(folder: str, extension: str) -> list:
    return [f for f in os.listdir(folder) if f.endswith(extension)]
