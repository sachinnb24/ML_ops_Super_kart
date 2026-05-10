from huggingface_hub import HfApi
import os


api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id="Sachinnb24/Super-kart-predition",
    repo_type="space",
    path_in_repo="",
)
