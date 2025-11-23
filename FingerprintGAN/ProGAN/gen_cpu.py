
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.insert(0, r'C:\Users\smart\Desktop\MassDataMining_project\FingerprintGAN\ProGAN')

import tensorflow as tf
import config
import tfutil
import misc
import numpy as np
from PIL import Image

# Initialize TensorFlow session for CPU
print('Initializing TensorFlow (CPU mode)...')
tfutil.init_tf(config.tf_config)

# Load generator
print('Loading network...')
G, D, Gs = misc.load_pkl('models/celeba_align_png_cropped.pkl')

# Generate images
latent_size = Gs.input_shape[1]  # Use Gs (generator for synthesis)
num_images = 5000
batch_size = 8  # Small batch for CPU

print(f'Generating {num_images} images...')
os.makedirs(r'C:\Users\smart\Desktop\MassDataMining_project\data\generated\celeba\progan', exist_ok=True)

for i in range(0, num_images, batch_size):
    current_batch = min(batch_size, num_images - i)
    latents = np.random.RandomState(0 + i).randn(current_batch, latent_size).astype(np.float32)
    
    # Generate with CPU (num_gpus must be 1, not 0)
    images = Gs.run(latents, None, minibatch_size=current_batch, num_gpus=1, 
                   out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
    
    # Save
    for j in range(len(images)):
        img = images[j]
        if img.ndim == 2:
            pil_img = Image.fromarray(img, 'L')
        else:
            pil_img = Image.fromarray(img, 'RGB')
        pil_img.save(r'C:\Users\smart\Desktop\MassDataMining_project\data\generated\celeba\progan/progan_{(i+j):05d}.png')
    
    if (i + current_batch) % 100 == 0:
        print(f'  Generated {i + current_batch}/{num_images}')

print('Done!')
