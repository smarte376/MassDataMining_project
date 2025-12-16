from pathlib import Path
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

RESULTS_DIR = "../results/"

def output_f1_score():
    results_root = Path(RESULTS_DIR)
    results_csv_path = next(results_root.rglob("results.csv"), None)

    if results_csv_path is None:
        raise ValueError("No classification results found. Run run_classification_pipeline.py and then ./parse_all.sh to generate results")

    df = pd.read_csv(results_csv_path, header=None)

    true_class = df.iloc[:, 1]
    predicted_class = df.iloc[:, 2]

    macro_f1_score = metrics.f1_score(true_class, predicted_class, labels=true_class.unique(), average="macro")
    print(f"Macro f1_score: {macro_f1_score:.4f}")

    classes = true_class.unique()
    f1_scores = metrics.f1_score(true_class, predicted_class, labels=classes, average=None)
    for class_, f1_score in zip(classes, f1_scores):
        print(f"{class_} f1 score: {f1_score:.4f}")

def output_auc_roc_score():
    results_root = Path(RESULTS_DIR)
    marker = "<" * 50 + "PROBABILITIES" + ">" * 50

    # found_probabilities = False
    for result_txt in results_root.rglob("classification_results/probabilities.txt"):
        with open(result_txt, "r") as file:
            true_classes = []
            probabilities = []
            label_line = next(file).split()[1]
            label_line_split = label_line.split(",")
            labels = [
                label_line_split[0].split("_")[1].lower(),
                label_line_split[1][:label_line_split[1].index("_")].lower(),
                label_line_split[2][:label_line_split[2].index("_")].lower(),
                label_line_split[3][:label_line_split[3].index("_")].lower(),
                label_line_split[4][:label_line_split[4].index("_")].lower()
            ]
            for line in file:
                data_line = line.split("|")
                image_file = data_line[0].split()[1]
                real_class = image_file[:image_file.index("_")]
                true_classes.append(real_class)
                probabilities.append(list(map(float, data_line[1].split()[1].split(","))))
            df = pd.DataFrame(data=probabilities, columns=labels)
            df.insert(0, "True Class", true_classes)

            plt.figure()
            roc_auc_scores = []
            for class_ in df["True Class"].unique():
                binary_class = df["True Class"] == class_
                scores = df[class_]
                roc_auc_score = metrics.roc_auc_score(
                    y_true = binary_class, 
                    y_score = scores,
                )
                roc_auc_scores.append((class_, roc_auc_score))
                fpr, tpr, _ = metrics.roc_curve(binary_class, scores)
                plt.plot(fpr, tpr, label=f"{class_} ROC (AUC={roc_auc_score:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
            plt.legend()
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve for All Classes")
            plt.savefig("roc_curve.png", format="png", bbox_inches="tight")
            found_probabilities = True

        if found_probabilities:
            break

def output_accuracy_score():
    results_root = Path(RESULTS_DIR)
    results_csv_path = next(results_root.rglob("results.csv"), None)

    df = pd.read_csv(results_csv_path, header=None)

    true_class = df.iloc[:, 1]
    predicted_class = df.iloc[:, 2]

    overall_accuracy_score = metrics.accuracy_score(true_class, predicted_class)
    print(f"Overall Accuracy: {overall_accuracy_score:.4f}")

    for class_ in true_class.unique():
        index_values = true_class[true_class == class_].index
        class_accuracy_score = metrics.accuracy_score(true_class[index_values], predicted_class[index_values])
        print(f"{class_} Accuracy: {class_accuracy_score:.4f}")

def main():
    output_f1_score()
    output_auc_roc_score()
    output_accuracy_score()   
        
if __name__ == "__main__":
    main()