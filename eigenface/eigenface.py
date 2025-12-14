from time import time

import matplotlib.pyplot as plt
from scipy.stats import loguniform
import regex as re

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import os
import os.path
import numpy as np
import PIL.Image
import skimage
import skimage.transform
import pickle
import pandas as pd


# based on https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html

def main():
    X, y = loadData(dataPath)
    n_features = X.shape[1]
    labels = np.array(['celeba', 'progan', 'sngan', 'cramergan', 'mmdgan'])
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

    n_components = 150

    print(
        "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
    )
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    save_pkl(pca, modelPath + "/ef_pca.pkl")

    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {
        "C": loguniform(1e3, 1e5),
        "gamma": loguniform(1e-4, 1e-1),
    }
    clf = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
    )
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    save_pkl(clf, modelPath + "/ef_classifier.pkl")

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    draw_cm(clf, X_test_pca, y_test, y_pred, labels)
    draw_eigenfaces(eigenfaces, h, w)
    
def draw_cm(clf, X_pca, y_test, y_pred, labels, save_dir="img"):
    X_test_pca = X_pca
    print(classification_report(y_test, y_pred, target_names=labels))
    if __name__ != "__main__":
        c_df = pd.DataFrame(np.array([y_test, y_pred]).T)
        c_df.to_csv(save_dir + "/eigenface_preds.csv")
        with open(save_dir + "/eigenface_report", "w") as f:
            f.write(classification_report(y_test, y_pred, target_names=labels))

    ConfusionMatrixDisplay.from_estimator(
        clf, X_test_pca, y_test, display_labels=labels, xticks_rotation="vertical"
    )
    plt.tight_layout()
    plt.savefig(save_dir + "/eigenface_cm.png", dpi=300)
    plt.show()

def draw_eigenfaces(eigenfaces, h, w):
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    plt.savefig("img/eigenfaces.png", dpi=300)
    plt.show()

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

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
        for (count0, name) in enumerate(name_list):
            label = re.match("(.*)_.*.(png|jpg)", name).group(1).lower()

            y.append(label)

            im = np.array(PIL.Image.open('%s/%s' % (testing_data_path, name)).convert('L')).astype(np.float32) / 255.0
            im = adjust_dynamic_range(im, [0,1], [-1,1])
            im = im.flatten()
            X.append(im)

        X = np.array(X)
        y = np.array(y)
        return X, y

def load_pkl(filename):
    with open(filename, 'rb') as file:
        # return legacy.LegacyUnpickler(file, encoding='latin1').load()
        return pickle.Unpickler(file, encoding='latin1').load()

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def run_model(dataPath, modelPath, savePath):
    X, y = loadData(dataPath)
    clf = load_pkl(modelPath + "/ef_classifier.pkl")
    pca = load_pkl(modelPath + "/ef_pca.pkl")
    labels = np.array(['celeba', 'progan', 'sngan', 'cramergan', 'mmdgan'])
    
    X_pca = pca.transform(X)
    y_pred = clf.predict(X_pca)
    draw_cm(clf, X_pca, y, y_pred, labels, savePath)

if __name__ == "__main__":
    dataPath = "../data/test_sets/celeba/eigenface_train/clean"
    modelPath = "models"

    main()