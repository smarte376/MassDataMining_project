from time import time

import matplotlib.pyplot as plt
import re

from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import os
import os.path
import numpy as np
import PIL.Image
import pickle
import pandas as pd

# Train on CelebA
# python knn.py --dataset celeba --data_path ../data/test_sets/celeba/knn_train/clean

# Train on LSUN bedroom
# python knn.py --dataset lsun_bedroom --data_path ../data/test_sets/lsun_bedroom/knn_train/clean

# Run inference on trained model
# python knn.py --mode inference --dataset celeba --data_path ../data/test_sets/celeba/balanced_small/clean
# k-Nearest Neighbors classifier for GAN-generated image detection

def main(dataset='celeba'):
    # Create necessary directories
    os.makedirs(modelPath, exist_ok=True)
    os.makedirs("img", exist_ok=True)
    
    X, y = loadData(dataPath)
    n_features = X.shape[1]
    labels = np.array(['real', 'progan', 'sngan', 'cramergan', 'mmdgan'])
    n_samples = X.shape[0]
    h, w = 128, 128
    n_classes = labels.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=84
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    print("Training kNN classifier on %d samples" % X_train.shape[0])
    t0 = time()
    
    # Grid search for optimal k
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'cosine']
    }
    
    clf = GridSearchCV(
        KNeighborsClassifier(), 
        param_grid, 
        cv=5, 
        n_jobs=1, #ran into an error using -1 with python 3.7, temp fix was to set it to 1, slow but works
        verbose=1
    )
    clf = clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print("Best parameters:")
    print(clf.best_params_)
    
    save_pkl(clf, modelPath + f"/knn_classifier_{dataset}.pkl")
    save_pkl(scaler, modelPath + f"/knn_scaler_{dataset}.pkl")

    print("Predicting on the test set")
    t0 = time()
    y_pred = clf.predict(X_test)
    print("done in %0.3fs" % (time() - t0))

    draw_cm(clf, X_test, y_test, y_pred, labels, save_dir="img", dataset=dataset)
    
def draw_cm(clf, X_test, y_test, y_pred, labels, save_dir="img", dataset="celeba"):
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save predictions and report
    c_df = pd.DataFrame(np.array([y_test, y_pred]).T, columns=['y_true', 'y_pred'])
    c_df.to_csv(save_dir + f"/knn_preds_{dataset}.csv", index=False)
    with open(save_dir + f"/knn_report_{dataset}.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, target_names=labels))

    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=labels, xticks_rotation="vertical"
    )
    plt.tight_layout()
    plt.savefig(save_dir + f"/knn_cm_{dataset}.png", dpi=300)
    plt.show()

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def loadData(testing_data_path):

    X = []
    y = []

    if os.path.isdir(testing_data_path):
        count_dict = None
        name_list = sorted(os.listdir(testing_data_path))
        length = len(name_list)
        print(f"Loading {length} images from {testing_data_path}")
        
        for (count0, name) in enumerate(name_list):
            # Show progress every 100 images
            if count0 % 100 == 0:
                print(f"  Progress: {count0}/{length} images loaded")
            
            try:
                label = re.match("(.*)_.*.(png|jpg)", name).group(1).lower()
                y.append(label)

                im = np.array(PIL.Image.open('%s/%s' % (testing_data_path, name)).convert('RGB')).astype(np.float32) / 255.0
                im = adjust_dynamic_range(im, [0,1], [-1,1])
                im = im.flatten()
                X.append(im)
            except Exception as e:
                print(f"  Warning: Failed to load {name}: {e}")
                continue
        
        print(f"  Completed: {len(X)}/{length} images loaded successfully")

        X = np.array(X)
        y = np.array(y)
        return X, y

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.Unpickler(file, encoding='latin1').load()

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def run_model(dataPath, modelPath, savePath, dataset='celeba'):
    X, y = loadData(dataPath)
    clf = load_pkl(modelPath + f"/knn_classifier_{dataset}.pkl")
    scaler = load_pkl(modelPath + f"/knn_scaler_{dataset}.pkl")
    labels = np.array(['real', 'progan', 'sngan', 'cramergan', 'mmdgan'])
    
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    draw_cm(clf, X_scaled, y, y_pred, labels, savePath, dataset)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='k-Nearest Neighbors classifier for GAN-generated image detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, default='celeba', 
                        choices=['celeba', 'lsun_bedroom'],
                        help='Dataset type (default: celeba)')
    parser.add_argument('--data_path', type=str, 
                        default='../data/test_sets/celeba/eigenface_train/clean',
                        help='Path to training/test data directory')
    parser.add_argument('--model_path', type=str, default='models',
                        help='Path to save/load models (default: models)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference'],
                        help='Mode: train new model or run inference (default: train)')
    parser.add_argument('--save_path', type=str, default='img',
                        help='Path to save results (default: img)')
    
    args = parser.parse_args()
    
    dataPath = args.data_path
    modelPath = args.model_path
    
    if args.mode == 'train':
        main(dataset=args.dataset)
    else:
        run_model(dataPath, modelPath, args.save_path, dataset=args.dataset)
