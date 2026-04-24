import os

def get_project_root() -> str:
    # Assuming this file is located at project_root/core/paths.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return project_root

def get_output_dir(sub_dir: str = "") -> str:
    """
    Returns a standardized output directory path within the project root.
    """
    path = os.path.join(get_project_root(), "outputs", sub_dir)
    os.makedirs(path, exist_ok=True)
    return path

def get_test_data_dir() -> str:
    """
    Returns the standardized test data directory path within the project root.
    """
    path = os.path.join(get_project_root(), "tests", "data")
    os.makedirs(path, exist_ok=True)
    return path

def get_test_output_dir(sub_dir: str = "") -> str:
    """
    Returns the standardized test output directory path within the project root.
    """
    path = os.path.join(get_project_root(), "tests", "outputs", sub_dir)
    os.makedirs(path, exist_ok=True)
    return path
