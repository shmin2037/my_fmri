import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

def data_transform(input_data, target, fname, slice):
    input_data = to_tensor(input_data)
    target = to_tensor(target)
    return input_data, target, fname, slice

# class DataTransform:
#     def __init__(self, input_data, target, fname, slice):
#         pass

#     def __call__(self, input_data, target, fname, slice):
#         input_data = to_tensor(input_data)
#         target = to_tensor(target)
# #         maximum = attrs.get('max', -1)  # Replace 'maximum_key' with the actual key name
        
#         return input_data, target, fname, slice
