# Configuration for CUDA 11.8 compatibility
import os

# Force specific GPU (or set to empty string for all GPUs)
gpu = '0'

# Environment variables
env = {
    'CUDA_VISIBLE_DEVICES': gpu,
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
    'TF_CPP_MIN_LOG_LEVEL': '1'
}

# TensorFlow config for CUDA 11.8
tf_config = {
    'gpu_options.allow_growth': True,
    'gpu_options.per_process_gpu_memory_fraction': 0.8,
    'allow_soft_placement': True,
    'log_device_placement': False
}

# Training
num_gpus = 1
minibatch_base = 16
minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
