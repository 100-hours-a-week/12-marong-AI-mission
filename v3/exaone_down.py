from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

HF_TOKEN = os.getenv("HF_TOKEN")

snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    local_dir="models/exaone-3.5-2.4b-instruct",
    local_dir_use_symlinks=False,
    token=HF_TOKEN
)