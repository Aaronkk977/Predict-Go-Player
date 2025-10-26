"""
Setup environment variables for stable training
Ensures all cache/temp files go to /tmp2 (large storage)
"""
import os
import pathlib

# Base directory for all temporary/cache files
BIG = "/tmp2/b12902115"

# Set environment variables
ENV_VARS = {
    "TMPDIR": f"{BIG}/tmp",
    "TORCH_HOME": f"{BIG}/torch-home",
    "TORCH_EXTENSIONS_DIR": f"{BIG}/torch_extensions",
    "CUDA_CACHE_PATH": f"{BIG}/cuda-cache",
    "HF_HOME": f"{BIG}/huggingface",
    "TRANSFORMERS_CACHE": f"{BIG}/hf-transformers",
    "HF_DATASETS_CACHE": f"{BIG}/hf-datasets",
}

def setup_environment():
    """Setup environment variables and create directories"""
    for key, value in ENV_VARS.items():
        os.environ.setdefault(key, value)
        pathlib.Path(value).mkdir(parents=True, exist_ok=True)
    
    print("Environment setup complete:")
    for key, value in ENV_VARS.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    setup_environment()
