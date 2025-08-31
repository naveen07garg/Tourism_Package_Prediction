from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "naveen07garg/Tourism-Package-Prediction"
repo_type = "dataset"

#== Local dir path where files are created to copy on HF ===
#project_dir="/content/drive/MyDrive/AIML_GreatLakes/MLOps/MLOps_Project/tourism_project/model_building"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

#=== Task 2 - Check if the space exists ===
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)
