
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import numpy as np
from PIL import Image
import yaml
import chainer

# Don't use GPU
chainer.config.use_ideep = 'never'

# Load config
with open('configs/sn_projection_celeba.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Import generator
sys.path.insert(0, 'gen_models')
import resnet

# Extract generator args from config
gen_args = config['models']['generator']['args']

# Create and load model with exact config from file
gen = resnet.ResNetGenerator(
    dim_z=gen_args['dim_z'],
    bottom_width=gen_args['bottom_width'],
    ch=gen_args['ch'],
    n_classes=gen_args['n_classes']  # Use config value (1) not 0
)
chainer.serializers.load_npz('models/celeba_align_png_cropped.npz', gen)

# Generate images
num_images = 5000
batch_size = 32
latent_size = gen_args['dim_z']

print(f'Generating {num_images} images...')
os.makedirs(r'C:\Users\smart\Desktop\MassDataMining_project\data\generated\celeba\sngan', exist_ok=True)

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    for i in range(0, num_images, batch_size):
        current_batch = min(batch_size, num_images - i)
        
        # Generate latent vectors
        z = np.random.randn(current_batch, latent_size).astype(np.float32)
        # Generate class labels (all class 0 for CelebA)
        y = np.zeros(current_batch, dtype=np.int32)
        
        # Generate images
        with chainer.using_config('train', False):
            x = gen(batchsize=current_batch, z=z, y=y)
            images = x.array
        
        # Convert to uint8
        images = np.clip((images + 1) * 127.5, 0, 255).astype(np.uint8)
        
        # Save images (NCHW to NHWC)
        for j in range(current_batch):
            img = np.transpose(images[j], (1, 2, 0))
            pil_img = Image.fromarray(img)
            pil_img.save(r'C:\Users\smart\Desktop\MassDataMining_project\data\generated\celeba\sngan/sngan_{(i+j):05d}.png')
        
        if (i + current_batch) % 100 == 0:
            print(f'  Generated {i + current_batch}/{num_images}')

print('Done!')
