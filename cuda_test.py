import torch
import xformers

def check_cuda_availability():
    """
    Check if CUDA is available and print relevant information.
    """
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Using CPU.")
        return False
    
def print_torch_version():
    """
    Print PyTorch version information.
    """
    print(f"PyTorch version: {torch.__version__}")
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

def check_xformers_availability():
    """
    Check if xFormers is available and print relevant information.
    """
    try:
        print("xFormers is available!")
        print(f"xFormers version: {xformers.__version__}")
        return True
    except ImportError:
        print("xFormers is not available.")
        return False
    except Exception as e:
        print(f"Error checking xFormers: {e}")
        return False

if __name__ == "__main__":
    check_cuda_availability()
    print_torch_version()
    check_xformers_availability()