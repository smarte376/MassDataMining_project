"""
Runs FingerprintGAN classifier on test sets with optional DefenseGAN defense
"""
#quick commands to test run file:
# FingerprintGAN classification only (no DefenseGAN)
# python run_classification_pipeline.py --test_set_dir data/test_sets/celeba/balanced_small/noise --dataset celeba   
# FingerprintGAN classification with DefenseGAN reconstruction
# python run_classification_pipeline.py --test_set_dir data/test_sets/celeba/balanced_small/noise --dataset celeba --use_defense_gan --defense_gan_model_dir DefenseGan/saved_model/collaborative_gan_cifar10_A               
import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Import DefenseGAN wrapper
from defense_gan_wrapper import DefenseGANReconstructor

from eigenface.eigenface import run_model

from knn.knn import run_model as run_knn_model


def run_classification_pipeline(
    test_set_dir,
    dataset,
    model_name,
    use_defense_gan=False,
    defense_gan_model_dir=None,
    defense_dataset_type='cifar10',
    output_base_dir='results',
    use_eigenface=False,
    eigenface_model_dir=None,
    use_knn=False,
    knn_model_dir=None
):
    """
    Args:
        test_set_dir: Directory containing test images
        dataset: Dataset name ('celeba' or 'lsun_bedroom')
        model_name: Name of the FingerprintGAN model to use
        use_defense_gan: Whether to apply DefenseGAN reconstruction
        defense_gan_model_dir: Path to DefenseGAN model directory (required if use_defense_gan=True)
        defense_dataset_type: DefenseGAN model type ('mnist', 'fmnist', 'cifar10')
        output_base_dir: Base directory for all output files
    """
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"{dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which test set to classify
    if use_defense_gan:
        if not defense_gan_model_dir:
            raise ValueError("defense_gan_model_dir must be provided when use_defense_gan=True")
        
        # Check if DefenseGAN model file exists
        defense_model_path = os.path.join(defense_gan_model_dir, 'g_ba.pth')
        if not os.path.exists(defense_model_path):
            print("=" * 80)
            print("WARNING: DefenseGAN Model Not Found")
            print("=" * 80)
            print(f"Expected model at: {defense_model_path}")
            print()
            print("For CelebA/LSUN (RGB images), you need a CIFAR-10 trained model (gen_ba_cf).")
            print("The model file should be saved as 'g_ba.pth' in your model directory.")
            print()
            print("Proceeding WITHOUT DefenseGAN reconstruction...")
            print("=" * 80)
            print()
            use_defense_gan = False
            classification_input_dir = str(Path(test_set_dir).absolute())
            defense_status = "no_defense"
        else:
            print("=" * 80)
            print("Step 1: Running DefenseGAN Reconstruction")
            print("=" * 80)
            
            # Create reconstructed test set directory
            reconstructed_dir = output_dir / "reconstructed_test_set"
            reconstructed_dir.mkdir(exist_ok=True)
            
            # Load DefenseGAN and reconstruct images
            defense_gan = DefenseGANReconstructor(
                model_path=defense_model_path,
                dataset_type=defense_dataset_type
            )
            
            defense_gan.reconstruct_directory(
                input_dir=test_set_dir,
                output_dir=str(reconstructed_dir),
                resize=(128, 128)  # FingerprintGAN expects 128x128
            )
            
            # Use reconstructed directory for classification (convert to absolute path)
            classification_input_dir = str(reconstructed_dir.absolute())
            defense_status = "with_defensegan"
        
    else:
        print("=" * 80)
        print("Skipping DefenseGAN Reconstruction (use_defense_gan=False)")
        print("=" * 80)
        
        # Use original test set for classification (convert to absolute path)
        classification_input_dir = str(Path(test_set_dir).absolute())
        defense_status = "no_defense"
    
    print()
    print("=" * 80)
    print("Step 2: Running FingerprintGAN Classification")
    print("=" * 80)
    
    # Create classification results directory
    results_dir = output_dir / "classification_results"
    results_dir.mkdir(exist_ok=True)
    
    # Prepare output filename (make absolute before changing directory)
    output_filename = f"{defense_status}_{Path(test_set_dir).name}_results.txt"
    output_path = (results_dir / output_filename).absolute()
    
    # Prepare FingerprintGAN classifier arguments
    classifier_dir = Path(__file__).parent.absolute() / "FingerprintGAN" / "classifier"
    
    print(f"Classifier directory: {classifier_dir}")
    print(f"Directory exists: {classifier_dir.exists()}")
    
    # Determine model path based on dataset
    if dataset.lower() == 'celeba':
        model_file = "CelebA_ProGAN_SNGAN_CramerGAN_MMDGAN_128.pkl"
    elif dataset.lower() == 'lsun_bedroom':
        model_file = "LSUN_bedroom_200k_ProGAN_SNGAN_CramerGAN_MMDGAN_128.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'celeba' or 'lsun_bedroom'")
    
    model_path = classifier_dir / "models" / model_file
    
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please ensure the FingerprintGAN model is downloaded and placed in the correct location.")
        return
    
    # Run FingerprintGAN classifier
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to classifier directory and add to Python path
        classifier_dir_str = str(classifier_dir.absolute())
        print(f"Changing to directory: {classifier_dir_str}")
        
        os.chdir(classifier_dir_str)
        sys.path.insert(0, classifier_dir_str)
        
        print(f"Current working directory: {os.getcwd()}")
        print(f"andrewm_scripts.py exists: {os.path.exists('andrewm_scripts.py')}")
        
        # Import classifier modules after changing directory
        import andrewm_scripts
        import tfutil
        
        # Initialize TensorFlow session
        tfutil.init_tf()
        
        print(f"Classifying images from: {classification_input_dir}")
        print(f"Using model: {model_path}")
        print(f"Saving results to: {output_path}")
        print()
        
        # Run classifier and capture output using a custom writer
        print("Running classification...")
        
        # Create a tee class to write to both stdout and file
        class TeeWriter:
            def __init__(self, file_path):
                self.terminal = sys.stdout
                self.log = open(file_path, 'w', encoding='utf-8')
                # Write header
                self.log.write("=" * 80 + "\n")
                self.log.write("FingerprintGAN Classification Results\n")
                self.log.write("=" * 80 + "\n\n")
                self.log.write(f"Model: {model_path}\n")
                self.log.write(f"Test Set: {classification_input_dir}\n")
                self.log.write(f"Defense Status: {defense_status}\n\n")
                self.log.write("=" * 80 + "\n")
                self.log.write("Classification Output\n")
                self.log.write("=" * 80 + "\n\n")
            
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
            
            def flush(self):
                self.terminal.flush()
                self.log.flush()
            
            def close(self):
                self.log.close()
        
        # Redirect stdout to tee writer
        tee = TeeWriter(str(output_path))
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            andrewm_scripts.classify(
                model_path=str(model_path),
                testing_data_path=classification_input_dir
            )
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            tee.close()
        
        print(f"\nClassification complete!")
        print(f"Results saved to: {output_path}")
        
    finally:
        # Return to original directory and clean up sys.path
        os.chdir(original_dir)
        if str(classifier_dir) in sys.path:
            sys.path.remove(str(classifier_dir))
    
    # Create summary
    summary_file = output_dir / "pipeline_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Classification Pipeline Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Original Test Set: {test_set_dir}\n")
        f.write(f"DefenseGAN Applied: {use_defense_gan}\n")
        if use_defense_gan and 'reconstructed_dir' in locals():
            f.write(f"DefenseGAN Model: {defense_gan_model_dir}\n")
            f.write(f"DefenseGAN Type: {defense_dataset_type}\n")
            f.write(f"Reconstructed Images: {reconstructed_dir}\n")
        f.write(f"Classification Input: {classification_input_dir}\n")
        f.write(f"Results Directory: {results_dir}\n")
        f.write(f"Predictions File: {output_path}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    if use_eigenface:
        if not eigenface_model_dir:
            raise ValueError("eigenface_model_dir must be provided when use_eigenface=True")

        print()
        print("=" * 80)
        print("Eigenface Classifier")
        print("=" * 80)
        run_model(test_set_dir, eigenface_model_dir, str(results_dir))
        print()
    else:
        print()
        print("=" * 80)
        print("Skipping Eigenface Classifier (use_eigenface=False)")
        print("=" * 80)

    if use_knn:
        if not knn_model_dir:
            raise ValueError("knn_model_dir must be provided when use_knn=True")

        print()
        print("=" * 80)
        print("kNN Classifier")
        print("=" * 80)
        run_knn_model(test_set_dir, knn_model_dir, str(results_dir), dataset=dataset)
        print()
    else:
        print()
        print("=" * 80)
        print("Skipping kNN Classifier (use_knn=False)")
        print("=" * 80)
    
    print()
    print("=" * 80)
    print("Pipeline Finished")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_file}")
    if use_defense_gan and 'reconstructed_dir' in locals():
        print(f"Reconstructed images: {reconstructed_dir}")
    print(f"Classification results: {results_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run FingerprintGAN classification with optional DefenseGAN reconstruction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run classification WITHOUT DefenseGAN
  python run_classification_pipeline.py \\
      --test_set_dir data/test_sets/celeba/balanced_small/clean \\
      --dataset celeba

  # Run classification WITH DefenseGAN reconstruction
  python run_classification_pipeline.py \\
      --test_set_dir data/test_sets/celeba/balanced_small/noise \\
      --dataset celeba \\
      --use_defense_gan \\
      --defense_gan_model_dir DefenseGan/saved_model/cifar10

  # Run classification WITH DefenseGAN reconstruction AND Eigenface classifier
  python run_classification_pipeline.py \\
      --test_set_dir data/test_sets/celeba/balanced_small/noise \\
      --dataset celeba \\
      --use_defense_gan \\
      --defense_gan_model_dir DefenseGan/saved_model/cifar10 \\
      --use_eigenface \\
      --eigenface_model_dir eigenface/models

  # Run on LSUN bedroom dataset
  python run_classification_pipeline.py \\
      --test_set_dir data/test_sets/lsun_bedroom/balanced_small/blur \\
      --dataset lsun_bedroom \\
      --use_defense_gan \\
      --defense_gan_model_dir DefenseGan/saved_model/cifar10
        """
    )
    
    parser.add_argument(
        '--test_set_dir',
        type=str,
        required=True,
        help='Directory containing test images to classify'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['celeba', 'lsun_bedroom'],
        help='Dataset type (determines which FingerprintGAN model to use)'
    )
    
    parser.add_argument(
        '--use_defense_gan',
        action='store_true',
        help='Apply DefenseGAN reconstruction before classification'
    )
    
    parser.add_argument(
        '--defense_gan_model_dir',
        type=str,
        default='DefenseGan/saved_model/collaborative_gan_cifar10_A',
        help='Path to DefenseGAN model directory (contains g_ba.pth). Note: MNIST model is grayscale only, need RGB model for CelebA/LSUN'
    )
    
    parser.add_argument(
        '--defense_dataset_type',
        type=str,
        default='cifar10',
        choices=['mnist', 'fmnist', 'cifar10'],
        help='DefenseGAN model type (default: cifar10 for 128x128 RGB images)'
    )
    
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='results',
        help='Base directory for output files (default: results/)'
    )

    parser.add_argument(
        '--use_eigenface',
        action='store_true',
        help='Use Eigenface classifier along with our native classifier; only works on CelebA'
    )

    parser.add_argument(
        '--eigenface_model_dir',
        type=str,
        default='eigenface/models',
        help='Path to Eigenface classifier directory; only works on CelebA'
    )

    parser.add_argument(
        '--use_knn',
        action='store_true',
        help='Use kNN classifier along with our native classifier'
    )

    parser.add_argument(
        '--knn_model_dir',
        type=str,
        default='knn/models',
        help='Path to kNN classifier directory'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.test_set_dir):
        print(f"ERROR: Test set directory not found: {args.test_set_dir}")
        sys.exit(1)
    
    if args.use_defense_gan and not os.path.isdir(args.defense_gan_model_dir):
        print(f"ERROR: DefenseGAN model directory not found: {args.defense_gan_model_dir}")
        sys.exit(1)

    if args.use_eigenface and not os.path.isdir(args.eigenface_model_dir):
        print(f"ERROR: Eigenface model directory not found: {args.eigenface_model_dir}")
        sys.exit(1)

    if args.use_knn and not os.path.isdir(args.knn_model_dir):
        print(f"ERROR: kNN model directory not found: {args.knn_model_dir}")
        sys.exit(1)
    
    # Run pipeline
    run_classification_pipeline(
        test_set_dir=args.test_set_dir,
        dataset=args.dataset,
        model_name=f"FingerprintGAN_{args.dataset}",
        use_defense_gan=args.use_defense_gan,
        defense_gan_model_dir=args.defense_gan_model_dir,
        defense_dataset_type=args.defense_dataset_type,
        output_base_dir=args.output_base_dir,
        use_eigenface=args.use_eigenface,
        eigenface_model_dir=args.eigenface_model_dir,
        use_knn=args.use_knn,
        knn_model_dir=args.knn_model_dir
    )


if __name__ == "__main__":
    main()
