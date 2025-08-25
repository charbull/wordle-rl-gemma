from huggingface_hub import HfApi, create_repo

if __name__ == "__main__":
    repo_id = "charbull/mlx_gemma3_4b_wordle_lora" # Replace with your username/repo_name
    adapter_path = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250824-133827_gemma-3-4b-it-bf16_rank64/adapters/adapter_step_500.npz"

    print(f"uploading adatper at file '{adapter_path}")

    # 2. Set up your repository information
    file_to_upload = adapter_path
    print("Initializing HfApi and starting upload...")
    api = HfApi()

    print(f"Ensuring repository '{repo_id}' exists...")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    print("Repository check complete.")

    info = api.upload_file(
        path_or_fileobj=file_to_upload,
        path_in_repo="wordle/adapter_final.npz", # The path inside the repository
        repo_id=repo_id,
        repo_type="model", # Can be "model", "dataset", or "space". Defaults to "model"
    )
