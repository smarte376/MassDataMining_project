"""
DefenseGAN Wrapper for Image Reconstruction
Integrates PyTorch-based DefenseGAN with the evaluation pipeline
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

# Add DefenseGan to path
DEFENSE_GAN_PATH = Path(__file__).parent / "DefenseGan"
sys.path.insert(0, str(DEFENSE_GAN_PATH))

from DefenseGan.models.disco_model import generator_ba, gen_ba_cf, generator_ab, gen_ab_cf


class DefenseGANReconstructor:
    """
    Wrapper for DefenseGAN reconstruction
    """
    
    def __init__(self, model_path, dataset_type='cifar10', device=None):
        """
        Args:
            model_path: Path to saved DefenseGAN model (g_ba.pth)
            dataset_type: 'mnist', 'fmnist', or 'cifar10'
            device: torch device (cuda/cpu)
        """
        self.dataset_type = dataset_type.lower()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the appropriate generator model
        if self.dataset_type == 'cifar10':
            self.g_ba = gen_ba_cf().to(self.device)
        else:
            self.g_ba = generator_ba().to(self.device)
        
        # Load weights
        if os.path.exists(model_path):
            print(f"Loading DefenseGAN model from {model_path}")
            self.g_ba.load_state_dict(torch.load(model_path, map_location=self.device))
            self.g_ba.eval()
        else:
            print(f"Warning: DefenseGAN model not found at {model_path}")
            print("Proceeding without defense...")
            self.g_ba = None
    
    def reconstruct_image(self, img_array):
        """
        Reconstruct a single image using DefenseGAN
        
        Args:
            img_array: numpy array (H, W, C) with values in [0, 1]
        
        Returns:
            Reconstructed image as numpy array (H, W, C)
        """
        if self.g_ba is None:
            return img_array
        
        with torch.no_grad():
            # Convert to tensor: (H, W, C) -> (1, C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            # Normalize to [-1, 1] for GAN
            img_tensor = img_tensor * 2.0 - 1.0
            
            # Reconstruct
            reconstructed = self.g_ba(img_tensor)
            
            # Denormalize back to [0, 1]
            reconstructed = (reconstructed + 1.0) / 2.0
            reconstructed = torch.clamp(reconstructed, 0.0, 1.0)
            
            # Convert back to numpy: (1, C, H, W) -> (H, W, C)
            reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return reconstructed_np
    
    
    
    def reconstruct_directory(self, input_dir, output_dir, resize=(128, 128)):
        """
        Reconstruct all images in a directory
        
        Args:
            input_dir: Input directory with images
            output_dir: Output directory for reconstructed images
            resize: Resize images to this size (for DefenseGAN compatibility)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Glob is case-insensitive on Windows, so lowercase patterns are enough
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(input_path.glob(ext)))
        
        print(f"Reconstructing {len(image_files)} images from {input_dir}...")
        
        for img_file in image_files:
            # Load image
            img = Image.open(img_file).convert('RGB')
            if resize:
                img = img.resize(resize)
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Reconstruct
            reconstructed = self.reconstruct_image(img_array)
            
            # Save
            reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
            reconstructed_pil = Image.fromarray(reconstructed_uint8)
            output_file = output_path / img_file.name
            reconstructed_pil.save(output_file)
        
        print(f"Saved reconstructed images to {output_dir}")


def load_defense_gan(model_dir, dataset_type='cifar10'):
    """
    Helper function to load DefenseGAN model
    
    Args:
        model_dir: Directory containing g_ba.pth
        dataset_type: 'mnist', 'fmnist', or 'cifar10'
    
    Returns:
        DefenseGANReconstructor instance
    """
    model_path = os.path.join(model_dir, 'g_ba.pth')
    return DefenseGANReconstructor(model_path, dataset_type)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='DefenseGAN Image Reconstruction')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to DefenseGAN model directory')
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for reconstructed images')
    parser.add_argument('--dataset_type', type=str, default='mnist', choices=['mnist', 'fmnist', 'cifar10'])
    parser.add_argument('--resize', type=int, nargs=2, default=[128, 128], help='Resize dimensions (H W)')
    
    args = parser.parse_args()
    
    # Load DefenseGAN
    defense_gan = load_defense_gan(args.model_dir, args.dataset_type)
    
    # Reconstruct directory
    defense_gan.reconstruct_directory(
        args.input_dir,
        args.output_dir,
        resize=tuple(args.resize)
    )
