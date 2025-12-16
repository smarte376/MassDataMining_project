"""
Test DefenseGAN reconstruction on adversarial examples.
This script demonstrates DefenseGAN's ability to defend against FGSM attacks
by reconstructing clean images from adversarial examples.

NOTE: This is a simplified demo version. For full DefenseGAN with trained models, 
you need to first run:
  1. python DefenseGAN/train_classifier.py --dataset mnist --arch B --n_epochs 10
  2. python DefenseGAN/train.py --dataset mnist --arch B --n_epochs 10
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Model paths (optional - will train simple models if not found)
CLF_MODEL_PATH = './saved_model/mnist_B.pth'
GAN_MODEL_PATH = './saved_model/collaborative_gan_mnist_B/g_ba.pth'

# ============================================================================
# Model Definitions
# ============================================================================

class mnistmodel_B(nn.Module):
    """MNIST Classifier - Architecture B"""
    def __init__(self):
        super(mnistmodel_B, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.dense1 = nn.Linear(in_features=128, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)
        return x

class generator_ba(nn.Module):
    """Generator for reconstructing clean images from adversarial examples"""
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=1 , out_channels=8, kernel_size=3,stride=1,padding=1)
        self.d2 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=3,stride=1,padding=1)
        self.d3 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=3,stride=1,padding=1)
        self.enmaxpool = nn.MaxPool2d(2)
        self.u1 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding=1)
        self.u2 = nn.Conv2d(in_channels=64,out_channels=16, kernel_size=3,padding=1)
        self.u3 = nn.Conv2d(in_channels=32,out_channels=8, kernel_size=3,padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.output = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1)
        
    def forward(self,x):
        d1 = F.leaky_relu(self.d1(x), 0.2)
        x = F.max_pool2d(d1,2)
        d2 = F.instance_norm(F.leaky_relu(self.d2(x), 0.2))
        x = F.max_pool2d(d2,2)
        d3 = F.instance_norm(F.leaky_relu(self.d3(x), 0.2))
        encoder = self.enmaxpool(d3)
        x = self.up1(encoder)
        x = nn.ZeroPad2d((1,0,1,0))(x)
        x = self.u1(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u1 = torch.cat((x,d3),1)
        x = self.up1(u1)
        x = self.u2(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u2 = torch.cat((x,d2),1)
        x  = self.up1(u2)
        x = self.u3(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u3 = torch.cat((x,d1),1)
        x = self.output(u3)
        x = F.relu(x)
        return x


class FGSM:
    """Fast Gradient Sign Method attack."""
    def __init__(self, model, eps=0.3):
        self.model = model
        self.eps = eps
    
    def perturb(self, x, y):
        """Generate adversarial example using FGSM."""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_adv)
        loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial example
        x_adv = x_adv + self.eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()

def load_mnist_data(train=False):
    """Load MNIST dataset."""
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    return dataset

def load_models():
    """Load or create the classifier and generator models."""
    clf_model = mnistmodel_B().to(device)
    g_ba = generator_ba().to(device)
    
    # Try to load pre-trained classifier
    if os.path.exists(CLF_MODEL_PATH):
        print("Loading pre-trained classifier...")
        clf_model.load_state_dict(torch.load(CLF_MODEL_PATH, map_location=device))
        clf_model.eval()
        print("✓ Classifier loaded successfully")
    else:
        print(f"WARNING: Classifier not found at {CLF_MODEL_PATH}")
        print("Training a quick classifier (this may take a few minutes)...")
        clf_model = train_quick_classifier()
        print("✓ Quick classifier trained")
    
    # Try to load pre-trained generator
    if os.path.exists(GAN_MODEL_PATH):
        print("Loading DefenseGAN generator...")
        g_ba.load_state_dict(torch.load(GAN_MODEL_PATH, map_location=device))
        g_ba.eval()
        print("✓ Generator loaded successfully")
    else:
        print(f"WARNING: Generator not found at {GAN_MODEL_PATH}")
        print("Using untrained generator (for demonstration only)")
        print("For proper DefenseGAN, run: python DefenseGAN/train.py --dataset mnist --arch B")
        g_ba.eval()
    
    return clf_model, g_ba

def train_quick_classifier():
    """Train a quick classifier for demonstration."""
    from tqdm import tqdm
    
    model = mnistmodel_B().to(device)
    train_dataset = load_mnist_data(train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(3):  # Just 3 epochs for quick demo
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/3')
        for batch_data, batch_target in pbar:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    model.eval()
    # Save the model
    os.makedirs(os.path.dirname(CLF_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), CLF_MODEL_PATH)
    print(f"✓ Model saved to {CLF_MODEL_PATH}")
    
    return model

def test_reconstruction(clf_model, g_ba, n=0, eps=0.3):
    """
    Test DefenseGAN reconstruction on a single adversarial example.
    
    Args:
        clf_model: Classifier model
        g_ba: Generator model for reconstruction
        n: Test sample index
        eps: FGSM attack strength
    """
    # Load test data
    test_dataset = load_mnist_data(train=False)
    
    # Get test image
    img, true_label = test_dataset[n]
    img = img.unsqueeze(0)  # Add batch dimension
    img_label = torch.tensor([true_label])
    
    # Generate adversarial example using FGSM
    img_atk = FGSM(clf_model, eps=eps).perturb(
        img.to(device), 
        img_label.to(device)
    ).cpu()
    
    # Reconstruct using DefenseGAN
    with torch.no_grad():
        img_reconstructed = g_ba(img_atk.to(device)).cpu()
        # Clip to valid range [0, 1] since generator_ba uses ReLU activation
        img_reconstructed = torch.clamp(img_reconstructed, 0, 1)
    
    # Get predictions
    with torch.no_grad():
        pred_clean = clf_model(img.to(device)).argmax(dim=1).item()
        pred_atk = clf_model(img_atk.to(device)).argmax(dim=1).item()
        # Use the clamped reconstruction for prediction
        pred_reconstructed = clf_model(img_reconstructed.to(device)).argmax(dim=1).item()
    
    # Print results
    print(f"\nTest sample {n}:")
    print(f"  True label: {true_label}")
    print(f"  Clean prediction: {pred_clean} {'✓' if pred_clean == true_label else '✗'}")
    print(f"  Attack prediction: {pred_atk} {'✓' if pred_atk == true_label else '✗'}")
    print(f"  Reconstructed prediction: {pred_reconstructed} {'✓' if pred_reconstructed == true_label else '✗'}")
    
    # Debug: Print value ranges
    print(f"\n  Image value ranges:")
    print(f"    Clean: [{img.min():.3f}, {img.max():.3f}]")
    print(f"    Adversarial: [{img_atk.min():.3f}, {img_atk.max():.3f}]")
    print(f"    Reconstructed: [{img_reconstructed.min():.3f}, {img_reconstructed.max():.3f}]")
    
    # Visualize results
    plt.gray()
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    
    # Clean image
    ax[0].imshow(img[0][0], vmin=0, vmax=1)
    ax[0].set_title(f'Clean Image\nPred: {pred_clean} (True: {true_label})')
    ax[0].axis('off')
    
    # Adversarial image
    ax[1].imshow(img_atk[0][0], vmin=0, vmax=1)
    ax[1].set_title(f'FGSM Attack (ε={eps})\nPred: {pred_atk}')
    ax[1].axis('off')
    
    # Reconstructed image
    ax[2].imshow(img_reconstructed[0][0], vmin=0, vmax=1)
    ax[2].set_title(f'DefenseGAN Reconstruction\nPred: {pred_reconstructed}')
    ax[2].axis('off')
    
    # Difference map
    diff = torch.abs(img - img_reconstructed)
    ax[3].imshow(diff[0][0], vmin=0, vmax=1, cmap='hot')
    ax[3].set_title(f'Reconstruction Error\nMAE: {diff.mean():.4f}')
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.show()


def test_multiple_epsilons(clf_model, g_ba, epsilon_values=[0.1, 0.2, 0.3, 0.4]):
    """
    Test DefenseGAN across different attack strengths.
    
    Args:
        clf_model: Classifier model
        g_ba: Generator model
        epsilon_values: List of epsilon values to test
    """
    print("\n" + "="*80)
    print("Testing DefenseGAN across multiple attack strengths")
    print("="*80)
    
    results = []
    
    # Load test data
    test_dataset = load_mnist_data(train=False)
    num_test_samples = min(100, len(test_dataset))
    
    for eps in epsilon_values:
        print(f"\nTesting with epsilon = {eps}")
        
        clean_correct = 0
        atk_correct = 0
        reconstructed_correct = 0
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        for i in range(num_test_samples):
            # Get random test image
            n = np.random.randint(0, len(test_dataset))
            img, true_label = test_dataset[n]
            img = img.unsqueeze(0)  # Add batch dimension
            img_label = torch.tensor([true_label])
            
            # Generate adversarial example
            img_atk = FGSM(clf_model, eps=eps).perturb(
                img.to(device), 
                img_label.to(device)
            )
            
            # Get predictions
            with torch.no_grad():
                pred_clean = clf_model(img.to(device)).argmax(dim=1).item()
                pred_atk = clf_model(img_atk).argmax(dim=1).item()
                
                # Reconstruct and clip to valid range
                img_reconstructed = g_ba(img_atk)
                img_reconstructed = torch.clamp(img_reconstructed, 0, 1)
                pred_reconstructed = clf_model(img_reconstructed).argmax(dim=1).item()
            
            if pred_clean == true_label:
                clean_correct += 1
            if pred_atk == true_label:
                atk_correct += 1
            if pred_reconstructed == true_label:
                reconstructed_correct += 1
        
        clean_acc = 100 * clean_correct / num_test_samples
        atk_acc = 100 * atk_correct / num_test_samples
        reconstructed_acc = 100 * reconstructed_correct / num_test_samples
        
        results.append({
            'epsilon': eps,
            'clean_acc': clean_acc,
            'attack_acc': atk_acc,
            'reconstructed_acc': reconstructed_acc
        })
        
        print(f"  Clean accuracy: {clean_acc:.1f}%")
        print(f"  Attack accuracy: {atk_acc:.1f}%")
        print(f"  Reconstructed accuracy: {reconstructed_acc:.1f}%")
        print(f"  Defense improvement: +{reconstructed_acc - atk_acc:.1f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Epsilon':<10} {'Clean Acc':<12} {'Attack Acc':<12} {'Defense Acc':<12} {'Improvement':<12}")
    print("-"*80)
    for r in results:
        improvement = r['reconstructed_acc'] - r['attack_acc']
        print(f"{r['epsilon']:<10.2f} {r['clean_acc']:<12.1f} {r['attack_acc']:<12.1f} "
              f"{r['reconstructed_acc']:<12.1f} +{improvement:<11.1f}")
    print("="*80)

def main():
    """Main function to test DefenseGAN."""
    print("="*80)
    print("DefenseGAN Reconstruction Test")
    print("="*80)
    
    # Load or train models
    clf_model, g_ba = load_models()
    
    # Check if generator is trained
    if not os.path.exists(GAN_MODEL_PATH):
        print("\n" + "!"*80)
        print("WARNING: Using UNTRAINED generator!")
        print("!"*80)
        print("The reconstruction will likely produce poor results.")
        print("For proper DefenseGAN defense, you need to train the generator first:")
        print("  python DefenseGAN/train.py --dataset mnist --arch B --n_epochs 10")
        print("\nThis demo will show the workflow, but expect random/noisy reconstructions.")
        print("="*80)
    
    print("\n" + "="*80)
    print("Running single sample visualization test")
    print("="*80)
    # Test reconstruction on a single sample
    test_reconstruction(clf_model, g_ba, n=0, eps=0.3)
    
    print("\n" + "="*80)
    print("Running quantitative evaluation across multiple attack strengths")
    print("="*80)
    # Test across multiple attack strengths
    test_multiple_epsilons(clf_model, g_ba, epsilon_values=[0.1, 0.2, 0.3, 0.4])
    
    print("\n✓ Testing complete!")


if __name__ == '__main__':
    main()
