"""Install inference dependencies on the Colab VM."""
import importlib.metadata
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vllm", "datasets"])
print("SETUP_OK", "vllm", importlib.metadata.version("vllm"), "torch", importlib.metadata.version("torch"))
