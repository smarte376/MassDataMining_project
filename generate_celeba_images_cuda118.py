"""
Generate CelebA images using all 4 GAN models with CUDA 11.8 support.
This script generates approximately 5000 images from each model:
- ProGAN
- SNGAN
- CramerGAN
- MMDGAN

Total: ~20,000 images
Compatible with TensorFlow 2.10.0 and CUDA 11.8 + cuDNN 8.9.7
"""

import os
import sys
import numpy as np
from pathlib import Path

# Fix Unicode encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

NUM_IMAGES = 5000
BATCH_SIZE_PROGAN = 16
BATCH_SIZE_SNGAN = 32
BATCH_SIZE_CRAMER = 64  # Checkpoint was trained with batch size 64
BATCH_SIZE_MMD = 128     # Checkpoint was trained with batch size 128
SEED = 0

# Output directories
OUTPUT_DIR_PROGAN = './data/generated/celeba/progan'
OUTPUT_DIR_SNGAN = './data/generated/celeba/sngan'
OUTPUT_DIR_CRAMER = './data/generated/celeba/cramergan'
OUTPUT_DIR_MMD = './data/generated/celeba/mmdgan'

# Model paths
PROGAN_MODEL = './FingerprintGAN/ProGAN/models/celeba_align_png_cropped.pkl'
SNGAN_MODEL = './FingerprintGAN/SNGAN/models/celeba_align_png_cropped.npz'
SNGAN_CONFIG = './FingerprintGAN/SNGAN/configs/sn_projection_celeba.yml'
CRAMER_MODEL = './FingerprintGAN/CramerGAN/models/celebA64x64_g_resnet5_dc_d5-5-1_32_128_lr0.00010000_bn'
MMD_MODEL = './FingerprintGAN/MMDGAN/models/celebA64x64_g_resnet5_dc_mix_rq_1dotd5-5-1_32_128_lr0.00010000_bn'

# ============================================================================
# ProGAN Generation
# ============================================================================

def generate_progan_images():
    """Generate images using ProGAN."""
    print("\n" + "="*80)
    print("GENERATING PROGAN IMAGES (CUDA 11.8)")
    print("="*80)
    print(f"Model: {PROGAN_MODEL}")
    print(f"Output: {OUTPUT_DIR_PROGAN}")
    print(f"Images: {NUM_IMAGES}")
    print()
    
    # Set environment for CUDA 11.8
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Import TensorFlow in compatibility mode
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    sys.path.insert(0, './FingerprintGAN/ProGAN')
    import misc
    
    # Create output directory
    os.makedirs(OUTPUT_DIR_PROGAN, exist_ok=True)
    
    # Initialize TensorFlow session
    print("Initializing TensorFlow...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    tf.set_random_seed(np.random.randint(1 << 31))
    sess = tf.Session(config=config)
    sess.__enter__()  # Make it the default session
    
    # Load generator
    print("Loading network...")
    G, D, Gs = misc.load_pkl(PROGAN_MODEL)
    
    # Generate images
    latent_size = Gs.input_shape[1]
    print(f"Latent size: {latent_size}")
    print(f"Generating {NUM_IMAGES} images...")
    
    for i in range(0, NUM_IMAGES, BATCH_SIZE_PROGAN):
        current_batch = min(BATCH_SIZE_PROGAN, NUM_IMAGES - i)
        
        # Generate latent vectors
        latents = np.random.RandomState(SEED + i).randn(current_batch, latent_size).astype(np.float32)
        # Create dummy labels (zeros) for unconditional generation
        labels = np.zeros([current_batch, 0], dtype=np.float32)
        
        # Generate images
        images = Gs.run(latents, labels,
                       minibatch_size=current_batch,
                       num_gpus=1,
                       out_mul=127.5, 
                       out_add=127.5, 
                       out_dtype=np.uint8)
        
        # Save images
        for j in range(len(images)):
            img = images[j]
            # ProGAN outputs in NCHW format (C, H, W), need to transpose to HWC (H, W, C) for PIL
            if img.ndim == 3 and img.shape[0] == 3:
                # Transpose from (C, H, W) to (H, W, C)
                img = img.transpose(1, 2, 0)
            if img.ndim == 2:
                pil_img = Image.fromarray(img, 'L')
            else:
                pil_img = Image.fromarray(img, 'RGB')
            pil_img.save(f'{OUTPUT_DIR_PROGAN}/progan_{(i+j):05d}.png')
        
        if (i + current_batch) % 100 == 0:
            print(f"  Generated {i + current_batch}/{NUM_IMAGES} images")
    
    print(f"\n[✓] ProGAN: Successfully generated {NUM_IMAGES} images to {OUTPUT_DIR_PROGAN}")
    
    # Clean up
    sess.close()

# ============================================================================
# SNGAN Generation
# ============================================================================

def generate_sngan_images():
    """Generate images using SNGAN."""
    print("\n" + "="*80)
    print("GENERATING SNGAN IMAGES (CUDA 11.8)")
    print("="*80)
    print(f"Model: {SNGAN_MODEL}")
    print(f"Output: {OUTPUT_DIR_SNGAN}")
    print(f"Images: {NUM_IMAGES}")
    print()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    sys.path.insert(0, './FingerprintGAN/SNGAN')
    sys.path.insert(0, './FingerprintGAN/SNGAN/gen_models')
    
    import yaml
    import chainer
    from chainer import serializers
    import resnet
    
    # Create output directory
    os.makedirs(OUTPUT_DIR_SNGAN, exist_ok=True)
    
    # Load config
    print("Loading configuration...")
    with open(SNGAN_CONFIG, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Extract generator args
    gen_args = config['models']['generator']['args']
    
    # Create model
    print("Creating generator...")
    gen = resnet.ResNetGenerator(
        dim_z=gen_args['dim_z'],
        bottom_width=gen_args['bottom_width'],
        ch=gen_args['ch'],
        n_classes=gen_args['n_classes']
    )
    
    # Use GPU
    print("Setting up GPU...")
    chainer.cuda.get_device_from_id(0).use()
    gen.to_gpu(0)
    
    # Load weights
    print("Loading weights...")
    serializers.load_npz(SNGAN_MODEL, gen)
    
    # Generate images
    latent_size = gen_args['dim_z']
    print(f"Latent size: {latent_size}")
    print(f"Generating {NUM_IMAGES} images...")
    
    np.random.seed(SEED)
    
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i in range(0, NUM_IMAGES, BATCH_SIZE_SNGAN):
            current_batch = min(BATCH_SIZE_SNGAN, NUM_IMAGES - i)
            
            # Generate latent vectors
            z = np.random.randn(current_batch, latent_size).astype(np.float32)
            z = chainer.cuda.to_gpu(z, 0)
            
            # Generate class labels (all class 0)
            y = np.zeros(current_batch, dtype=np.int32)
            y = chainer.cuda.to_gpu(y, 0)
            
            # Generate images
            with chainer.using_config('train', False):
                x = gen(batchsize=current_batch, z=z, y=y)
                images = chainer.cuda.to_cpu(x.array)
            
            # Convert to uint8
            images = np.clip((images + 1) * 127.5, 0, 255).astype(np.uint8)
            
            # Save images (NCHW to NHWC)
            for j in range(current_batch):
                img = np.transpose(images[j], (1, 2, 0))
                pil_img = Image.fromarray(img)
                pil_img.save(f'{OUTPUT_DIR_SNGAN}/sngan_{(i+j):05d}.png')
            
            if (i + current_batch) % 100 == 0:
                print(f"  Generated {i + current_batch}/{NUM_IMAGES} images")
    
    print(f"\n[✓] SNGAN: Successfully generated {NUM_IMAGES} images to {OUTPUT_DIR_SNGAN}")

# ============================================================================
# CramerGAN Generation
# ============================================================================

def generate_cramergan_images():
    """Generate images using CramerGAN."""
    print("\n" + "="*80)
    print("GENERATING CRAMERGAN IMAGES (CUDA 11.8)")
    print("="*80)
    print(f"Model: {CRAMER_MODEL}")
    print(f"Output: {OUTPUT_DIR_CRAMER}")
    print(f"Images: {NUM_IMAGES}")
    print()
    
    # Set environment for CUDA 11.8
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Import TensorFlow in compatibility mode
    import tensorflow as tf
    if hasattr(tf, 'compat'):
        tf = tf.compat.v1
        tf.disable_v2_behavior()
    
    sys.path.insert(0, str(Path(__file__).parent / 'FingerprintGAN' / 'CramerGAN' / 'gan'))
    
    # Create output directory
    os.makedirs(OUTPUT_DIR_CRAMER, exist_ok=True)
    
    # Create FLAGS
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    # Clear existing flags if any
    for key in list(FLAGS.__flags.keys()):
        delattr(FLAGS, key)
    FLAGS.__dict__['__flags'] = {}
    FLAGS.__dict__['__parsed'] = False
    
    # Define required flags
    flags.DEFINE_string("checkpoint_dir", CRAMER_MODEL, "Checkpoint directory")
    flags.DEFINE_string("dataset", "celebA", "Dataset name")
    flags.DEFINE_integer("batch_size", BATCH_SIZE_CRAMER, "Batch size")
    flags.DEFINE_integer(
"output_size", 64, "Output size")
    flags.DEFINE_integer("c_dim", 3, "Image channels")
    flags.DEFINE_string("architecture", "g_resnet5", "Architecture")
    flags.DEFINE_string("model", "cramer", "Model type")
    flags.DEFINE_integer("gf_dim", 64, "Generator filters")  # Checkpoint: g_h0_lin expects 64*16*4*4=16384
    flags.DEFINE_integer("df_dim", 64, "Discriminator filters")  # Model trained with df_dim=64
    flags.DEFINE_integer("dof_dim", 256, "Discriminator output features")  # Checkpoint: d_h6_lin (16384, 256)
    flags.DEFINE_float("gradient_penalty", 1.0, "Gradient penalty")
    flags.DEFINE_boolean("batch_norm", True, "Use batch norm")
    flags.DEFINE_boolean("is_train", False, "Training mode")
    flags.DEFINE_string("data_dir", "./data", "Data directory")
    flags.DEFINE_integer("dsteps", 5, "D steps")
    flags.DEFINE_integer("gsteps", 1, "G steps")
    flags.DEFINE_integer("start_dsteps", 5, "Start D steps")
    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
    flags.DEFINE_float("learning_rate_D", 0.0001, "D learning rate")
    flags.DEFINE_string("output_dir_of_test_samples", "", "Output directory")
    flags.DEFINE_integer("max_iteration", 150000, "Max iterations")
    flags.DEFINE_boolean("MMD_lr_scheduler", False, "MMD LR scheduler")
    flags.DEFINE_float("decay_rate", 0.5, "Decay rate")
    flags.DEFINE_float("gp_decay_rate", 1.0, "GP decay rate")
    flags.DEFINE_float("beta1", 0.5, "Beta1")
    flags.DEFINE_float("init", 0.1, "Init")
    flags.DEFINE_integer("real_batch_size", BATCH_SIZE_CRAMER, "Real batch size")
    flags.DEFINE_string("name", "cramer_test", "Name")
    flags.DEFINE_string("sample_dir", "samples", "Sample dir")
    flags.DEFINE_string("log_dir", "logs", "Log dir")
    flags.DEFINE_string("kernel", "", "Kernel")
    flags.DEFINE_boolean("visualize", False, "Visualize")
    flags.DEFINE_boolean("is_demo", False, "Demo")
    flags.DEFINE_integer("threads", 64, "Threads")
    flags.DEFINE_boolean("log", False, "Log")
    flags.DEFINE_string("suffix", '', "Suffix")
    flags.DEFINE_boolean("compute_scores", False, "Compute scores")
    flags.DEFINE_float("gpu_mem", 0.8, "GPU memory")
    flags.DEFINE_float("L2_discriminator_penalty", 0.0, "L2 penalty")
    flags.DEFINE_integer("no_of_samples", NUM_IMAGES, "Number of samples")
    flags.DEFINE_boolean("print_pca", False, "Print PCA")
    flags.DEFINE_integer("save_layer_outputs", 0, "Save layer outputs")
    flags.DEFINE_integer("random_seed", 0, "Random seed")
    FLAGS.__dict__['__parsed'] = True
    
    # Create session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    with tf.Session(config=sess_config) as sess:
        # Import and create model
        from core.cramer import Cramer_GAN
        
        # Create model
        gan = Cramer_GAN(
            sess=sess,
            config=FLAGS,
            batch_size=BATCH_SIZE_CRAMER,
            output_size=128,  # Checkpoint trained with 128x128 output (gf_dim*16*s32*s32=64*16*4*4=16384)
            c_dim=3,
            data_dir=FLAGS.data_dir
        )
        
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        # Load checkpoint
        print(f"Loading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(CRAMER_MODEL)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            gan.saver.restore(sess, os.path.join(CRAMER_MODEL, ckpt_name))
            print(f"Successfully loaded checkpoint: {ckpt_name}")
        else:
            print("[!] Failed to load checkpoint")
            return
        
        print("Setting up random z placeholder for image generation...")
        sys.stdout.flush()
        
        # Create a placeholder for random z input (avoids discriminator computation)
        z_input = tf.placeholder(tf.float32, [None, 100], name='z_random')
        
        # Get the generator architecture
        from core.architecture import get_networks
        Generator, _ = get_networks(FLAGS.architecture)
        generator_func = Generator(gan.gf_dim, gan.c_dim, gan.output_size, FLAGS.batch_norm)
        
        # Create a new generator graph that reuses trained weights
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            batch_size_tensor = tf.shape(z_input)[0]
            G_random = generator_func(z_input, batch_size_tensor)
        
        print(f"Generator ready. Generating {NUM_IMAGES} images...")
        sys.stdout.flush()
        
        # Generate images in batches
        num_batches = (NUM_IMAGES + BATCH_SIZE_CRAMER - 1) // BATCH_SIZE_CRAMER
        generated_count = 0
        
        from scipy.ndimage import zoom
        
        for batch_idx in range(num_batches):
            # Determine batch size for this iteration
            remaining = NUM_IMAGES - generated_count
            current_batch_size = min(BATCH_SIZE_CRAMER, remaining)
            
            # Generate random z values
            z = np.random.uniform(-1, 1, size=(current_batch_size, 100)).astype(np.float32)
            
            # Generate images using our random z placeholder
            images = sess.run(G_random, feed_dict={z_input: z})
            
            # Process and save each image
            for i in range(current_batch_size):
                img = images[i]
                
                # Clip to [0, 1] and convert to uint8 (output is already 128x128)
                img = np.clip(img, 0, 1)
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Save image
                image_path = os.path.join(OUTPUT_DIR_CRAMER, f'cramergan_{generated_count:05d}.png')
                Image.fromarray(img_uint8).save(image_path)
                
                generated_count += 1
            
            # Show progress after each batch
            if (batch_idx + 1) % 5 == 0 or generated_count >= NUM_IMAGES:
                print(f"  Generated {generated_count}/{NUM_IMAGES} images (batch {batch_idx+1}/{num_batches})")
                sys.stdout.flush()
    
    print(f"\n[✓] CramerGAN: Successfully generated {generated_count} images to {OUTPUT_DIR_CRAMER}")
    
    # Clean up session
    sess.close()
    
    # Clean up
    tf.reset_default_graph()

# ============================================================================
# MMDGAN Generation
# ============================================================================

def generate_mmdgan_images():
    """Generate images using MMDGAN."""
    print("\n" + "="*80)
    print("GENERATING MMDGAN IMAGES (CUDA 11.8)")
    print("="*80)
    print(f"Model: {MMD_MODEL}")
    print(f"Output: {OUTPUT_DIR_MMD}")
    print(f"Images: {NUM_IMAGES}")
    print()
    
    # Import sys first before any usage
    import sys
    
    # Set environment for CUDA 11.8
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Import TensorFlow in compatibility mode
    import tensorflow as tf
    if hasattr(tf, 'compat'):
        tf = tf.compat.v1
        tf.disable_v2_behavior()
    
    sys.path.insert(0, str(Path(__file__).parent / 'FingerprintGAN' / 'MMDGAN' / 'gan'))
    
    # Create output directory
    os.makedirs(OUTPUT_DIR_MMD, exist_ok=True)
    
    # Create FLAGS
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    # Clear existing flags if any
    for key in list(FLAGS.__flags.keys()):
        delattr(FLAGS, key)
    FLAGS.__dict__['__flags'] = {}
    FLAGS.__dict__['__parsed'] = False
    
    # Define required flags
    flags.DEFINE_string("checkpoint_dir", MMD_MODEL, "Checkpoint directory")
    flags.DEFINE_string("dataset", "celebA", "Dataset name")
    flags.DEFINE_integer("batch_size", BATCH_SIZE_MMD, "Batch size")
    flags.DEFINE_integer("output_size", 64, "Output size")
    flags.DEFINE_integer("c_dim", 3, "Image channels")
    flags.DEFINE_string("architecture", "g_resnet5_mix_rq", "Architecture")
    flags.DEFINE_string("model", "mmd", "Model type")
    flags.DEFINE_string("kernel", "mix_rq_1dot", "Kernel type")
    flags.DEFINE_integer("gf_dim", 64, "Generator filters")  # Checkpoint: g_h0_lin expects 64*16*4*4=16384
    flags.DEFINE_integer("df_dim", 64, "Discriminator filters")  # Model trained with df_dim=64
    flags.DEFINE_integer("dof_dim", 16, "Discriminator output features")  # Checkpoint: d_h6_lin (16384, 16)
    flags.DEFINE_float("gradient_penalty", 1.0, "Gradient penalty")
    flags.DEFINE_boolean("batch_norm", True, "Use batch norm")
    flags.DEFINE_boolean("is_train", False, "Training mode")
    flags.DEFINE_string("data_dir", "./data", "Data directory")
    flags.DEFINE_integer("dsteps", 5, "D steps")
    flags.DEFINE_integer("gsteps", 1, "G steps")
    flags.DEFINE_integer("start_dsteps", 5, "Start D steps")
    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
    flags.DEFINE_float("learning_rate_D", 0.0001, "D learning rate")
    flags.DEFINE_string("output_dir_of_test_samples", "", "Output directory")
    flags.DEFINE_integer("max_iteration", 150000, "Max iterations")
    flags.DEFINE_boolean("MMD_lr_scheduler", False, "MMD LR scheduler")
    flags.DEFINE_float("decay_rate", 0.5, "Decay rate")
    flags.DEFINE_float("gp_decay_rate", 1.0, "GP decay rate")
    flags.DEFINE_float("beta1", 0.5, "Beta1")
    flags.DEFINE_float("init", 0.1, "Init")
    flags.DEFINE_integer("real_batch_size", BATCH_SIZE_MMD, "Real batch size")
    flags.DEFINE_string("name", "mmd_test", "Name")
    flags.DEFINE_string("sample_dir", "samples", "Sample dir")
    flags.DEFINE_string("log_dir", "logs", "Log dir")
    flags.DEFINE_boolean("visualize", False, "Visualize")
    flags.DEFINE_boolean("is_demo", False, "Demo")
    flags.DEFINE_integer("threads", 64, "Threads")
    flags.DEFINE_boolean("log", False, "Log")
    flags.DEFINE_string("suffix", '', "Suffix")
    flags.DEFINE_boolean("compute_scores", False, "Compute scores")
    flags.DEFINE_float("gpu_mem", 0.8, "GPU memory")
    flags.DEFINE_float("L2_discriminator_penalty", 0.0, "L2 penalty")
    flags.DEFINE_integer("no_of_samples", NUM_IMAGES, "Number of samples")
    flags.DEFINE_boolean("print_pca", False, "Print PCA")
    flags.DEFINE_integer("save_layer_outputs", 0, "Save layer outputs")
    flags.DEFINE_integer('random_seed', 0, 'Random seed')
    FLAGS.__dict__['__parsed'] = True
    
    # Note: Graph reset happens in main() between CramerGAN and MMDGAN
    # Do NOT reset here as it causes nested graph errors
    
    # Create session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    with tf.Session(config=sess_config) as sess:
        # Import and create model
        from core.model import MMD_GAN
        
        # Create model  
        gan = MMD_GAN(
            sess=sess,
            config=FLAGS,
            batch_size=BATCH_SIZE_MMD,
            output_size=128,  # Checkpoint trained with 128x128 output (gf_dim*16*s32*s32=64*16*4*4=16384)
            c_dim=3,
            data_dir=FLAGS.data_dir
        )
        
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        # Load checkpoint
        print(f"Loading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(MMD_MODEL)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            gan.saver.restore(sess, os.path.join(MMD_MODEL, ckpt_name))
            print(f"Successfully loaded checkpoint: {ckpt_name}")
        else:
            print("[!] Failed to load checkpoint")
            return
        
        print("Setting up random z placeholder for image generation...")
        sys.stdout.flush()
        
        # Create a placeholder for random z input (avoids discriminator computation)
        z_input = tf.placeholder(tf.float32, [None, 100], name='z_random')
        
        # Get the generator architecture
        from core.architecture import get_networks
        Generator, _ = get_networks(FLAGS.architecture)
        generator_func = Generator(gan.gf_dim, gan.c_dim, gan.output_size, FLAGS.batch_norm)
        
        # Create a new generator graph that reuses trained weights
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            G_random = generator_func(z_input, tf.shape(z_input)[0])
        
        print(f"Generator ready. Generating {NUM_IMAGES} images...")
        sys.stdout.flush()
        
        # Generate images in batches
        num_batches = (NUM_IMAGES + BATCH_SIZE_MMD - 1) // BATCH_SIZE_MMD
        generated_count = 0
        
        from scipy.ndimage import zoom
        
        for batch_idx in range(num_batches):
            # Determine batch size for this iteration
            remaining = NUM_IMAGES - generated_count
            current_batch_size = min(BATCH_SIZE_MMD, remaining)
            
            # Generate random z values
            z = np.random.uniform(-1, 1, size=(current_batch_size, 100)).astype(np.float32)
            
            # Generate images using our random z placeholder
            images = sess.run(G_random, feed_dict={z_input: z})
            
            # Process and save each image
            for i in range(current_batch_size):
                img = images[i]
                
                # Clip to [0, 1] and convert to uint8 (output is already 128x128)
                img = np.clip(img, 0, 1)
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Save image
                image_path = os.path.join(OUTPUT_DIR_MMD, f'mmdgan_{generated_count:05d}.png')
                Image.fromarray(img_uint8).save(image_path)
                
                generated_count += 1
            
            # Show progress after each batch
            if (batch_idx + 1) % 5 == 0 or generated_count >= NUM_IMAGES:
                print(f"  Generated {generated_count}/{NUM_IMAGES} images (batch {batch_idx+1}/{num_batches})")
                sys.stdout.flush()
    
    print(f"\n[✓] MMDGAN: Successfully generated {generated_count} images to {OUTPUT_DIR_MMD}")
    
    # Clean up
    tf.reset_default_graph()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to generate all images."""
    print("="*80)
    print("CELEBA IMAGE GENERATION - CUDA 11.8")
    print("="*80)
    print("Generating images from 4 GAN models:")
    print("  - ProGAN (TensorFlow)")
    print("  - SNGAN (Chainer)")
    print("  - CramerGAN (TensorFlow)")
    print("  - MMDGAN (TensorFlow)")
    print(f"\nTarget: {NUM_IMAGES} images per model")
    print("Total: ~20,000 images")
    print("="*80)
    
    results = {}
    
    # Generate from each model
    # try:
    #     generate_progan_images()
    #     results['ProGAN'] = True
    # except Exception as e:
    #     print(f"\n[✗] ProGAN failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     results['ProGAN'] = False
    
    # try:
    #     generate_sngan_images()
    #     results['SNGAN'] = True
    # except Exception as e:
    #     print(f"\n[✗] SNGAN failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     results['SNGAN'] = False
    
    # try:
    #     generate_cramergan_images()
    #     results['CramerGAN'] = True
    # except Exception as e:
    #     print(f"\n[✗] CramerGAN failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     results['CramerGAN'] = False
    
    # IMPORTANT: CramerGAN session is now closed, safe to reset graph
    # Clean up TensorFlow graph before MMDGAN (only needed if CramerGAN ran)
    # print("\nCleaning up TensorFlow graph before MMDGAN...")
    # import tensorflow as tf
    # import time
    # import gc
    # 
    # # Force garbage collection
    # gc.collect()
    # 
    # # Reset default graph (now safe since session is closed)
    # if hasattr(tf, 'compat'):
    #     tf.compat.v1.reset_default_graph()
    # else:
    #     tf.reset_default_graph()
    #     
    # # Clear module cache to prevent graph conflicts  
    # modules_to_clear = [k for k in sys.modules.keys() if 'core' in k or 'architecture' in k]
    # for mod in modules_to_clear:
    #     try:
    #         del sys.modules[mod]
    #     except:
    #         pass
    #     
    # # Brief pause to ensure cleanup completes
    # time.sleep(1)
    # print("Graph cleanup complete.")
    
    try:
        generate_mmdgan_images()
        results['MMDGAN'] = True
    except Exception as e:
        print(f"\n[✗] MMDGAN failed: {e}")
        import traceback
        traceback.print_exc()
        results['MMDGAN'] = False
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY - CelebA")
    print("="*80)
    
    for model_name, success in results.items():
        status = "[✓] SUCCESS" if success else "[✗] FAILED"
        print(f"{status}: {model_name}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nTotal: {successful}/{total} models completed successfully")
    
    if successful == total:
        print("\n[✓] All CelebA images generated successfully!")
        print("Expected output:")
        print(f"  - {OUTPUT_DIR_PROGAN}/     ({NUM_IMAGES} images)")
        print(f"  - {OUTPUT_DIR_SNGAN}/      ({NUM_IMAGES} images)")
        print(f"  - {OUTPUT_DIR_CRAMER}/     ({NUM_IMAGES} images)")
        print(f"  - {OUTPUT_DIR_MMD}/        ({NUM_IMAGES} images)")
        return 0
    else:
        print("\n[!] Some models failed. Check the output above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
