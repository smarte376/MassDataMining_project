from pathlib import Path
import pandas as pd
from sklearn import metrics

RESULTS_DIR = "../results/"

def main():
    results_root = Path(RESULTS_DIR)
    results_csv_path = next(results_root.rglob("results.csv"), None)

    if results_csv_path is None:
        raise ValueError("No classification results found. Run run_classification_pipeline.py to generate results")

    df = pd.read_csv(results_csv_path, header=None)

    true_class = df.iloc[:, 1]
    predicted_class = df.iloc[:, 2]

    f1_score = metrics.f1_score(true_class, predicted_class, average="macro")
    print(f"f1_score: {f1_score}")

    
        
if __name__ == "__main__":
    main()