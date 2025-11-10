import tensorflow as tf
import os
import sys

def check_cuda_paths():
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64'
    ]
    
    path = os.environ.get('PATH', '').split(';')
    for cuda_path in cuda_paths:
        if cuda_path in path:
            print(f"Found in PATH: {cuda_path}")
        else:
            print(f"Missing from PATH: {cuda_path}")

def check_cudnn():
    cudnn_files = [
        (r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\cudnn64_7.dll', 'cuDNN DLL'),
        (r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include\cudnn.h', 'cuDNN Header'),
        (r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64\cudnn.lib', 'cuDNN Lib')
    ]
    
    for file_path, description in cudnn_files:
        if os.path.exists(file_path):
            print(f"{description} found: {file_path}")
        else:
            print(f"{description} missing: {file_path}")

print("Python version:", sys.version)
print("\nTensorFlow version:", tf.__version__)
print("\nChecking CUDA paths in environment:")
check_cuda_paths()

print("\nChecking cuDNN files:")
check_cudnn()

print("\nTensorFlow GPU configuration:")
from tensorflow.python.client import device_lib
print("\nAvailable devices:")
print(device_lib.list_local_devices())

print("\nIs built with CUDA:", tf.test.is_built_with_cuda())
print("GPU device name:", tf.test.gpu_device_name())

# Try to perform a simple GPU operation
try:
    # Create a TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            # Create some tensors
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
            # Multiply them
            c = tf.matmul(a, b)
            # Run the operation
            result = sess.run(c)
            print("\nGPU Computation test:")
            print("Matrix multiplication result:")
            print(result)
            print("GPU test successful!")
except Exception as e:
    print("\nGPU Computation test failed:")
    print(str(e))