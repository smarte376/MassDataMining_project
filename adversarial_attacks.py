"""
Adversarial Attack Generator
Creates adversarially attacked images for testing defense mechanisms
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import os
import cv2
from io import BytesIO


class AdversarialAttackGenerator:
    """
    Generates adversarial examples using various attack methods
    """
    
    def __init__(self, attack_type='noise', device=None, **kwargs):
        """
        Args:
            attack_type: 'noise', 'blur', 'crop', 'jpeg', 'relight', 'random_combo'
            device: torch device
            **kwargs: Override parameters if needed
        """
        self.attack_type = attack_type.lower()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def noise_attack(self, images):
        """
        Gaussian noise attack
        Adds i.i.d. Gaussian noise with variance randomly sampled from U[5.0, 20.0]
        
        Args:
            images: Input images (numpy array or torch tensor)
        """
        # Randomly sample variance from U[5.0, 20.0]
        variance = np.random.uniform(5.0, 20.0)
        std = np.sqrt(variance)
        
        is_tensor = isinstance(images, torch.Tensor)
        if is_tensor:
            images_np = images.cpu().numpy()
        else:
            images_np = images
        
        # Add Gaussian noise (std is in [0, 255] scale)
        # Convert to [0, 255] scale for noise application
        images_scaled = images_np * 255.0
        noise = np.random.normal(0, std, images_scaled.shape)
        noisy_images = images_scaled + noise
        
        # Clip and convert back to [0, 1]
        noisy_images = np.clip(noisy_images, 0, 255) / 255.0
        
        if is_tensor:
            return torch.from_numpy(noisy_images.astype(np.float32)).to(images.device)
        return noisy_images.astype(np.float32)
    
    def blur_attack(self, images):
        """
        Gaussian blur attack
        Performs Gaussian filtering with kernel size randomly picked from {1, 3, 5, 7, 9}
        
        Args:
            images: Input images (numpy array N x H x W x C)
        """
        # Randomly pick kernel size from {1, 3, 5, 7, 9}
        kernel_size = np.random.choice([1, 3, 5, 7, 9])
        
        blurred_images = []
        for img in images:
            if kernel_size == 1:
                # No blur for kernel size 1
                blurred_images.append(img)
            else:
                # Convert to uint8 for OpenCV
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Apply Gaussian blur using OpenCV
                blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
                
                # Convert back to float32 [0, 1]
                blurred_np = blurred.astype(np.float32) / 255.0
                blurred_images.append(blurred_np)
        
        return np.array(blurred_images)
    
    def crop_attack(self, images):
        """
        Cropping attack
        Crops images with random offset between 5% and 20% of side lengths, then resizes back
        
        Args:
            images: Input images (numpy array N x H x W x C)
        """
        cropped_images = []
        for img in images:
            h, w = img.shape[:2]
            
            # Random offset between 5% and 20%
            offset_percent = np.random.uniform(0.05, 0.20)
            
            # Calculate crop dimensions (keep 1 - 2*offset of the image)
            crop_h = int(h * (1 - 2 * offset_percent))
            crop_w = int(w * (1 - 2 * offset_percent))
            
            # Random starting position within the offset range
            start_h = int(np.random.uniform(offset_percent * h, h - crop_h - offset_percent * h))
            start_w = int(np.random.uniform(offset_percent * w, w - crop_w - offset_percent * w))
            
            # Ensure valid bounds
            start_h = max(0, min(start_h, h - crop_h))
            start_w = max(0, min(start_w, w - crop_w))
            
            # Crop
            cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Resize back to original size
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            cropped_images.append(resized)
        
        return np.array(cropped_images)
    
    def jpeg_compression_attack(self, images):
        """
        JPEG compression attack
        Quality factor randomly sampled from U[10, 75]
        
        Args:
            images: Input images (numpy array N x H x W x C)
        """
        # Randomly sample quality from U[10, 75]
        quality = int(np.random.uniform(10, 75))
        
        compressed_images = []
        for img in images:
            # Convert to PIL
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            
            # Compress using JPEG
            buffer = BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer)
            
            # Convert back to numpy
            compressed_np = np.array(compressed).astype(np.float32) / 255.0
            compressed_images.append(compressed_np)
        
        return np.array(compressed_images)
    
    def relighting_attack(self, images):
        """
        Relighting attack
        Simplified version: randomly adjusts brightness and contrast
        For full implementation, use SfSNet [54] to replace lighting conditions
        
        Args:
            images: Input images (numpy array N x H x W x C)
        """
        relighted_images = []
        for img in images:
            # Convert to PIL
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            
            # Random brightness factor [0.7, 1.5]
            brightness_factor = np.random.uniform(0.7, 1.5)
            enhancer = ImageEnhance.Brightness(pil_img)
            brightened = enhancer.enhance(brightness_factor)
            
            # Random contrast factor [0.8, 1.3]
            contrast_factor = np.random.uniform(0.8, 1.3)
            enhancer = ImageEnhance.Contrast(brightened)
            relighted = enhancer.enhance(contrast_factor)
            
            # Convert back to numpy
            relighted_np = np.array(relighted).astype(np.float32) / 255.0
            relighted_images.append(relighted_np)
        
        return np.array(relighted_images)
    
    def random_combo_attack(self, images):
        """
        Apply random combination of transformations
        Each attack is applied with 50% probability in order:
        relighting -> cropping -> blur -> JPEG compression -> noise
        
        Args:
            images: Input images (numpy array N x H x W x C)
        """
        result = images.copy()
        applied = []
        
        # Order: relighting, cropping, blur, JPEG, noise
        # Each with 50% probability
        
        if np.random.random() < 0.5:
            result = self.relighting_attack(result)
            applied.append('relight')
        
        if np.random.random() < 0.5:
            result = self.crop_attack(result)
            applied.append('crop')
        
        if np.random.random() < 0.5:
            result = self.blur_attack(result)
            applied.append('blur')
        
        if np.random.random() < 0.5:
            result = self.jpeg_compression_attack(result)
            applied.append('jpeg')
        
        if np.random.random() < 0.5:
            result = self.noise_attack(result)
            applied.append('noise')
        
        if len(applied) > 0:
            print(f"  Applied combo: {' -> '.join(applied)}")
        else:
            print(f"  Applied combo: none (all skipped)")
        
        return result
    
    def generate_adversarial_examples(self, images, labels=None):
        """
        Generate adversarial examples based on attack type
        
        Args:
            images: Input images numpy array (N, H, W, C) or tensor (N, C, H, W)
            labels: Not used for transformation attacks (kept for compatibility)
        
        Returns:
            adversarial_images: Perturbed images numpy array or tensor
        """
        # Convert tensor to numpy if needed
        is_tensor = isinstance(images, torch.Tensor)
        if is_tensor:
            images_np = images.cpu().numpy()
            # Convert from (N, C, H, W) to (N, H, W, C)
            if images_np.shape[1] in [1, 3]:
                images_np = np.transpose(images_np, (0, 2, 3, 1))
        else:
            images_np = images
        
        # Apply transformation attack
        if self.attack_type == 'noise':
            result = self.noise_attack(images_np)
        elif self.attack_type == 'blur':
            result = self.blur_attack(images_np)
        elif self.attack_type == 'crop':
            result = self.crop_attack(images_np)
        elif self.attack_type == 'jpeg':
            result = self.jpeg_compression_attack(images_np)
        elif self.attack_type == 'relight':
            result = self.relighting_attack(images_np)
        elif self.attack_type == 'random_combo':
            result = self.random_combo_attack(images_np)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            # Convert from (N, H, W, C) to (N, C, H, W)
            result = np.transpose(result, (0, 3, 1, 2))
            result = torch.from_numpy(result).to(images.device)
        
        return result
    
    def save_adversarial_dataset(self, data_loader, output_dir, class_names=None):
        """
        Generate and save adversarial examples from a data loader
        
        Args:
            data_loader: PyTorch DataLoader
            output_dir: Output directory
            class_names: List of class names (optional)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        img_idx = 0
        
        for batch_idx, (images, labels) in enumerate(data_loader):
            # Generate adversarial examples
            adv_images = self.generate_adversarial_examples(images)
            
            # Save each image
            for i in range(adv_images.shape[0]):
                adv_img = adv_images[i].cpu().numpy()
                
                # Convert from (C, H, W) to (H, W, C)
                if adv_img.shape[0] in [1, 3]:
                    adv_img = np.transpose(adv_img, (1, 2, 0))
                
                # Convert to uint8
                adv_img = (adv_img * 255).astype(np.uint8)
                
                # Convert to PIL Image
                if adv_img.shape[2] == 1:
                    adv_img = adv_img.squeeze(2)
                    pil_img = Image.fromarray(adv_img, mode='L')
                else:
                    pil_img = Image.fromarray(adv_img, mode='RGB')
                
                # Save with informative filename
                label = labels[i].item()
                if class_names:
                    label_name = class_names[label]
                else:
                    label_name = f"class_{label}"
                
                filename = f"{self.attack_type}_eps{self.epsilon}_{label_name}_{img_idx:05d}.png"
                pil_img.save(os.path.join(output_dir, filename))
                img_idx += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches, {img_idx} images...")
        
        print(f"Saved {img_idx} adversarial examples to {output_dir}")


def create_adversarial_attack_generator(attack_type='fgsm', epsilon=0.3, **kwargs):
    """
    Factory function to create attack generator
    """
    return AdversarialAttackGenerator(attack_type=attack_type, epsilon=epsilon, **kwargs)


# Standalone attack generation (no model required, uses numpy)
def generate_random_noise_attack(images, epsilon=0.1):
    """
    Generate random noise attack (baseline)
    
    Args:
        images: numpy array (N, H, W, C) in [0, 1]
        epsilon: Noise magnitude
    
    Returns:
        attacked_images: numpy array with noise
    """
    noise = np.random.uniform(-epsilon, epsilon, images.shape)
    attacked = np.clip(images + noise, 0, 1)
    return attacked


def save_numpy_images(images, output_dir, prefix="img"):
    """
    Save numpy images to directory
    
    Args:
        images: numpy array (N, H, W, C) or (N, H, W)
        output_dir: Output directory
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Create PIL Image
        if len(img.shape) == 2 or img.shape[2] == 1:
            if len(img.shape) == 3:
                img_uint8 = img_uint8.squeeze(2)
            pil_img = Image.fromarray(img_uint8, mode='L')
        else:
            pil_img = Image.fromarray(img_uint8, mode='RGB')
        
        # Save
        filename = f"{prefix}_{i:05d}.png"
        pil_img.save(os.path.join(output_dir, filename))
    
    print(f"Saved {len(images)} images to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Adversarial Examples')
    parser.add_argument('--attack_type', type=str, default='noise', 
                       choices=['noise', 'blur', 'crop', 'jpeg', 'relight', 'random_combo'],
                       help='Attack type')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Attack generator configured: {args.attack_type}, epsilon={args.epsilon}")
    print(f"Use this module in your pipeline to generate attacks")
