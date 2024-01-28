import torch

print("torch version:",torch.__version__)
print("torch cuda version",torch.version.cuda)
print("torch cuda is abailable:",torch.cuda.is_available())

import torch.utils
import torch.utils.cpp_extension as ex

print("CUDA_HOME:",ex.CUDA_HOME)