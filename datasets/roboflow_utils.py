import os
from roboflow import Roboflow

def download_roboflow_dataset(workspace_name: str, project_name: str, version_number: str = "1", overwrite: bool = False) -> str:
    """
    Download a dataset from Roboflow.
    
    Args:
        workspace_name: Name of the Roboflow workspace
        project_name: Name of the project
        version_number: Version number of the dataset (default: "1")
        overwrite: Whether to overwrite existing dataset (default: False)
        
    Returns:
        str: Path to the downloaded dataset
    """
    try:
        api_key = "3KKkPOIsCcLZX3PhX7xo"
        base_dir = os.path.dirname(os.path.abspath(__file__))

        print(f"URL: https://universe.roboflow.com/{workspace_name}/{project_name}")

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_name).project(project_name)
        output_dir = os.path.join(base_dir, f"{workspace_name}_{project_name}")

        dataset = project.version(version_number).download("yolov11", output_dir, overwrite=overwrite)
        print(f"Dataset downloaded successfully to: {dataset.location}")
        print(f"Number of images: {len(os.listdir(os.path.join(dataset.location, 'train/images')))}")
        
        return dataset.location
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise 