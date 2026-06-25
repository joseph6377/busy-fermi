import os
import sys
import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def get_hf_token():
    # Attempt to read token from .env
    env_path = Path(".env")
    if not env_path.exists():
        print("Error: .env file not found")
        return None
    
    token = None
    with open(env_path, "r") as f:
        for line in f:
            m = re.match(r"^HF_TOKEN\s*=\s*(.+)$", line.strip())
            if m:
                token = m.group(1).strip().strip("'\"")
                break
    return token

def main():
    token = get_hf_token()
    if not token:
        print("Error: HF_TOKEN is missing or not set in .env")
        sys.exit(1)
        
    api = HfApi(token=token)
    try:
        whoami = api.whoami()
        username = whoami["name"]
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Failed to authenticate with Hugging Face token: {e}")
        print("Please verify that your HF_TOKEN in .env is valid.")
        sys.exit(1)
        
    repo_id = f"{username}/flux-2-klein-9b-consolidated"
    print(f"Preparing private repository: {repo_id}")
    
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=True,
            repo_type="model",
            exist_ok=True
        )
        print("Repository is ready (created or already exists).")
    except Exception as e:
        print(f"Failed to create repository {repo_id}: {e}")
        print("Make sure your Hugging Face token has WRITE permissions (role: write).")
        sys.exit(1)
        
    local_dir = Path("local_models")
    if not local_dir.exists() or not local_dir.is_dir():
        print("Error: local_models directory not found. Please run npm run download:flux-klein first.")
        sys.exit(1)
        
    print(f"Uploading files from '{local_dir}' to '{repo_id}'...")
    try:
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[".gitkeep"],
            # Using multi_commits allows large uploads to be split into multiple PRs/commits,
            # avoiding gateway timeouts for extremely large files on standard git push
            multi_commits=True,
            multi_commits_verbose=True
        )
        print("Model consolidation upload complete!")
        print(f"Your private model repository: https://huggingface.co/models/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
