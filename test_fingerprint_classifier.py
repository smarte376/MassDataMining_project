"""
Test FingerprintGAN Classifier
Tests the pre-trained classifier on test set images to verify it's working correctly
before running the full pipeline.
"""
# Example usage:
# # Test CelebA classifier on clean images
# python test_fingerprint_classifier.py --dataset celeba --test_set balanced_small --attack clean

# # Test CelebA classifier on noise-attacked images
# python test_fingerprint_classifier.py --dataset celeba --test_set balanced_small --attack noise

# # Test LSUN bedroom classifier on clean images
# python test_fingerprint_classifier.py --dataset lsun_bedroom --test_set balanced_small --attack clean

# # Test LSUN bedroom classifier with blur attack
# python test_fingerprint_classifier.py --dataset lsun_bedroom --test_set balanced_small --attack blur

import os
import sys
import argparse

def test_classifier(dataset='celeba', test_set='balanced_small', attack_type='clean'):
    """
    Test the FingerprintGAN classifier on a specific test set
    
    Args:
        dataset: 'celeba' or 'lsun_bedroom'
        test_set: Which test set configuration to use (e.g., 'balanced_small')
        attack_type: 'clean' or attack type (e.g., 'noise', 'blur', 'crop', etc.)
    """
    
    # Determine paths based on dataset
    if dataset == 'celeba':
        model_path = './FingerprintGAN/classifier/models/CelebA_ProGAN_SNGAN_CramerGAN_MMDGAN_128.pkl'
        testing_data_path = f'./data/test_sets/celeba/{test_set}/{attack_type}'
    elif dataset == 'lsun_bedroom':
        model_path = './FingerprintGAN/classifier/models/LSUN_bedroom_200k_ProGAN_SNGAN_CramerGAN_MMDGAN_128.pkl'
        testing_data_path = f'./data/test_sets/lsun_bedroom/{test_set}/{attack_type}'
    else:
        print(f"Error: Unknown dataset '{dataset}'. Use 'celeba' or 'lsun_bedroom'")
        return
    
    # Verify paths exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(testing_data_path):
        print(f"Error: Test data not found at {testing_data_path}")
        print(f"Please run the test set creation script first:")
        if dataset == 'celeba':
            print("  python create_celeba_test_sets.py")
        else:
            print("  python create_lsun_bedroom_test_sets.py")
        return
    
    # Change to classifier directory
    original_dir = os.getcwd()
    classifier_dir = './FingerprintGAN/classifier'
    os.chdir(classifier_dir)
    
    try:
        # Build command
        cmd = f'python run.py --app test --model_path {os.path.join("..", "..", model_path)} --testing_data_path {os.path.join("..", "..", testing_data_path)}'
        
        print("=" * 80)
        print(f"TESTING FINGERPRINTGAN CLASSIFIER")
        print("=" * 80)
        print(f"Dataset: {dataset}")
        print(f"Model: {model_path}")
        print(f"Test set: {test_set}")
        print(f"Attack type: {attack_type}")
        print(f"Testing data: {testing_data_path}")
        print("\n" + "=" * 80)
        print(f"Running: {cmd}")
        print("=" * 80 + "\n")
        
        # Run the command
        os.system(cmd)
        
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test FingerprintGAN Classifier')
    parser.add_argument('--dataset', type=str, default='celeba', 
                       choices=['celeba', 'lsun_bedroom'],
                       help='Dataset to test (celeba or lsun_bedroom)')
    parser.add_argument('--test_set', type=str, default='balanced_small',
                       help='Test set configuration to use')
    parser.add_argument('--attack', type=str, default='clean',
                       choices=['clean', 'noise', 'blur', 'crop', 'jpeg', 'relight', 'random_combo'],
                       help='Attack type to test')
    
    args = parser.parse_args()
    
    test_classifier(args.dataset, args.test_set, args.attack)


if __name__ == "__main__":
    main()
