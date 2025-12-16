"""
Create Test Sets for LSUN Bedroom Dataset
Randomly selects images from real and generated image sets, then creates:
1. Clean test set (no perturbations)
2. Test sets with each type of adversarial attack applied
"""
# # Create all test sets (clean + all attacks)
# python create_lsun_bedroom_test_sets.py

# # Only clean test sets
# python create_lsun_bedroom_test_sets.py --clean_only

# # Only attacked test sets
# python create_lsun_bedroom_test_sets.py --attacks_only

import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path
import argparse
from PIL import Image

# Import adversarial attack generator
from adversarial_attacks import AdversarialAttackGenerator

# Configuration
REAL_IMAGE_DIR = './Datasets/Datasets/lsun_bedroom_train_200k_png'  # Real LSUN bedroom images
GENERATED_BASE_DIR = './data/generated/lsun_bedroom'
OUTPUT_BASE_DIR = './data/test_sets/lsun_bedroom'

# GAN types
GAN_TYPES = ['progan', 'sngan', 'cramergan', 'mmdgan']

# Attack types to test
ATTACK_TYPES = ['noise', 'blur', 'crop', 'jpeg', 'relight', 'random_combo']

# Test set configurations
TEST_CONFIGS = {
    'knn_train': {
        'real': 500,
        'progan': 500,
        'sngan': 500,
        'cramergan': 500,
        'mmdgan': 500
    }
}


def get_image_files(directory):
    """Get all image files from directory"""
    if not os.path.exists(directory):
        return []
    
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(Path(directory).glob(ext)))
        image_files.extend(list(Path(directory).glob(ext.upper())))
    
    return sorted(image_files)


def load_image_as_numpy(img_path):
    """Load image as numpy array in [0, 1] range"""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array


def save_numpy_as_image(img_array, output_path):
    """Save numpy array as image"""
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    img.save(output_path)


def apply_attack_to_directory(source_dir, output_dir, attack_type):
    """
    Apply adversarial attack to all images in a directory
    
    Args:
        source_dir: Source directory with clean images
        output_dir: Output directory for attacked images
        attack_type: Type of attack to apply
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    image_files = get_image_files(source_dir)
    if len(image_files) == 0:
        return 0
    
    # Create attack generator
    attack_gen = AdversarialAttackGenerator(attack_type=attack_type)
    
    print(f"    Applying {attack_type} attack to {len(image_files)} images...")
    
    for img_path in image_files:
        # Load image
        img = load_image_as_numpy(img_path)
        img_batch = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Apply attack
        attacked_batch = attack_gen.generate_adversarial_examples(img_batch)
        attacked_img = attacked_batch[0]  # Remove batch dimension
        
        # Save attacked image with same filename
        output_path = os.path.join(output_dir, img_path.name)
        save_numpy_as_image(attacked_img, output_path)
    
    return len(image_files)


def create_test_set_with_attack(test_name, config, real_dir, generated_base_dir, output_base_dir, attack_type=None, seed=42):
    """
    Create a flat test set by randomly selecting images and optionally applying an attack
    All images are saved in a single directory with labeled filenames
    
    Args:
        test_name: Name of the test set
        config: Dictionary with counts for each category
        real_dir: Directory with real images
        generated_base_dir: Base directory with generated images
        output_base_dir: Output directory for test set
        attack_type: Type of attack to apply (None for clean dataset)
        seed: Random seed for reproducibility
    """
    attack_suffix = attack_type if attack_type else 'clean'
    print(f"\n{'='*80}")
    print(f"Creating test set: {test_name} ({attack_suffix})")
    print(f"{'='*80}")
    
    random.seed(seed)
    output_dir = os.path.join(output_base_dir, test_name, attack_suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    total_copied = 0
    attack_gen = AdversarialAttackGenerator(attack_type=attack_type) if attack_type else None
    
    for category, count in config.items():
        print(f"\n[{category}] Selecting {count} images...")
        
        # Determine source directory
        if category == 'real':
            source_dir = real_dir
        else:
            source_dir = os.path.join(generated_base_dir, category)
        
        # Get available images
        available_images = get_image_files(source_dir)
        
        if len(available_images) == 0:
            print(f"  ⚠ No images found in {source_dir}")
            continue
        
        if len(available_images) < count:
            print(f"  ⚠ Only {len(available_images)} images available (requested {count})")
            count = len(available_images)
        
        # Randomly select images
        selected_images = random.sample(available_images, count)
        
        # Save images directly to flat directory
        for i, img_path in enumerate(selected_images):
            # Filename with category prefix (all in same directory)
            new_filename = f"{category}_{i:05d}{img_path.suffix}"
            dest_path = os.path.join(output_dir, new_filename)
            
            if attack_type:
                # Load, attack, and save
                img = load_image_as_numpy(img_path)
                img_batch = np.expand_dims(img, axis=0)
                attacked_batch = attack_gen.generate_adversarial_examples(img_batch)
                attacked_img = attacked_batch[0]
                save_numpy_as_image(attacked_img, dest_path)
            else:
                # Just copy
                shutil.copy2(img_path, dest_path)
        
        if attack_type:
            print(f"  ✓ Applied {attack_type} attack to {len(selected_images)} images -> {output_dir}")
        else:
            print(f"  ✓ Copied {len(selected_images)} images to {output_dir}")
        total_copied += len(selected_images)
    
    print(f"\n✓ Test set '{test_name}' ({attack_suffix}) created:")
    print(f"  - Flat structure: {output_dir}")
    print(f"  - Total images: {total_copied}")
    
    return total_copied, output_dir


def main():
    parser = argparse.ArgumentParser(description='Create LSUN Bedroom Test Sets with Adversarial Attacks')
    parser.add_argument('--real_dir', type=str, default=REAL_IMAGE_DIR, 
                       help='Directory with real LSUN bedroom images')
    parser.add_argument('--generated_dir', type=str, default=GENERATED_BASE_DIR,
                       help='Base directory with generated images')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_DIR,
                       help='Output directory for test sets')
    parser.add_argument('--test_sets', type=str, nargs='+', 
                       help='Specific test sets to create (default: all)')
    parser.add_argument('--attacks', type=str, nargs='+', default=ATTACK_TYPES,
                       help='Attack types to apply')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--clean_only', action='store_true',
                       help='Only create clean test sets without attacks')
    parser.add_argument('--attacks_only', action='store_true',
                       help='Only create attacked test sets with new random samples (skip clean)')
    parser.add_argument('--same_samples', action='store_true',
                       help='Use same random samples for all attacks (default: each attack gets different samples)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CREATING LSUN BEDROOM TEST SETS WITH ADVERSARIAL ATTACKS")
    print("=" * 80)
    print(f"Real images: {args.real_dir}")
    print(f"Generated images: {args.generated_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Attacks: {', '.join(args.attacks)}")
    if args.clean_only:
        print("Mode: Clean test sets only (no attacks)")
    elif args.attacks_only:
        print("Mode: Apply attacks to existing clean test sets only")
    
    # Verify directories exist
    if not os.path.exists(args.real_dir):
        print(f"\n⚠ Warning: Real image directory not found: {args.real_dir}")
        print("Please update the REAL_IMAGE_DIR path in the script or use --real_dir")
        return
    
    if not os.path.exists(args.generated_dir):
        print(f"\n⚠ Warning: Generated image directory not found: {args.generated_dir}")
        print("Please run the generation scripts first")
        return
    
    # Determine which test sets to create
    if args.test_sets:
        test_sets_to_create = {k: v for k, v in TEST_CONFIGS.items() if k in args.test_sets}
    else:
        test_sets_to_create = TEST_CONFIGS
    
    # Create test sets (each attack gets its own random selection)
    total_images = 0
    test_set_info = {}
    
    if not args.attacks_only:
        # Create clean test set
        for test_name, config in test_sets_to_create.items():
            images_copied, clean_dir = create_test_set_with_attack(
                test_name, 
                config, 
                args.real_dir, 
                args.generated_dir, 
                args.output_dir,
                attack_type=None,  # Clean dataset
                seed=args.seed
            )
            total_images += images_copied
            
            if test_name not in test_set_info:
                test_set_info[test_name] = {}
            test_set_info[test_name]['clean'] = images_copied
        
        # Create attacked test sets (each with its own random selection)
        if not args.clean_only:
            for attack_type in args.attacks:
                for test_name, config in test_sets_to_create.items():
                    # Use same seed or different seed based on flag
                    if args.same_samples:
                        attack_seed = args.seed  # Same samples as clean
                    else:
                        attack_seed = args.seed + hash(attack_type) % 10000  # Different samples
                    
                    images_copied, attack_dir = create_test_set_with_attack(
                        test_name, 
                        config, 
                        args.real_dir, 
                        args.generated_dir, 
                        args.output_dir,
                        attack_type=attack_type,
                        seed=attack_seed
                    )
                    total_images += images_copied
                    test_set_info[test_name][attack_type] = images_copied
    else:
        # In attacks_only mode, create only attacked versions with new random samples
        print("\n[Attacks Only Mode] Creating attacked test sets with new random samples...")
        for attack_type in args.attacks:
            for test_name, config in test_sets_to_create.items():
                attack_seed = args.seed + hash(attack_type) % 10000
                images_copied, attack_dir = create_test_set_with_attack(
                    test_name, 
                    config, 
                    args.real_dir, 
                    args.generated_dir, 
                    args.output_dir,
                    attack_type=attack_type,
                    seed=attack_seed
                )
                total_images += images_copied
                
                if test_name not in test_set_info:
                    test_set_info[test_name] = {}
                test_set_info[test_name][attack_type] = images_copied
    
    print("\n" + "=" * 80)
    print("TEST SET CREATION COMPLETE!")
    print("=" * 80)
    print(f"Processed {len(test_set_info)} test set configurations")
    print(f"Total images across all datasets: {total_images}")
    
    print(f"\nTest sets created:")
    for test_name, datasets in test_set_info.items():
        total_for_config = sum(datasets.values())
        print(f"\n  {test_name}: {total_for_config} total images across {len(datasets)} datasets")
        for dataset_type, count in datasets.items():
            print(f"    └─ {dataset_type}: {count} images")
    
    if args.clean_only:
        print(f"\nNote: Only clean datasets created (--clean_only mode)")
    elif args.attacks_only:
        print(f"\nNote: Only attacked datasets created with new random samples (--attacks_only mode)")
    else:
        print(f"\nNote: Each attack type has its own randomly sampled dataset")


if __name__ == "__main__":
    main()
