import torch


def print_gpu_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        # Provide guidance for enabling GPU in Colab
        print("⚠️  No GPU available. This notebook requires a GPU for efficient training.")


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory cached: {cached:.2f} GB")
    else:
        print("⚠️  No GPU available. Cannot report GPU memory usage.")


def clean_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared GPU memory cache.")
    else:
        print("⚠️  No GPU available. Cannot clear GPU memory cache.")

