from pathlib import Path
import pandas as pd

RESULTS_DIR = "../results/"

def read_until_results_line(file):
    target_line = "Loading network..."
    for line in file:
        if target_line in line:
            return
    raise ValueError("Cannot find line of results")

def main():
    results_root = Path(RESULTS_DIR)
    results_txt_path = next(results_root.rglob("classification_results/*.txt"), None)
    if results_txt_path is None:
        raise ValueError("No classification results found. Run run_classification_pipeline.py to generate results")

    image_names = []
    image_classifications = []

    with open(Path(results_txt_path), "r") as file:
        read_until_results_line(file)
        for line in file:
            line_split = line.split()

            if line_split[0] != "Classifying":
                break
            
            image_names.append(line_split[3][:-1]) # Remove the trailing colon
            image_classifications.append(line_split[-1])

    df = pd.DataFrame(data={
        "name": image_names,
        "classified_as": image_classifications
    })

    print(df.head())
        
if __name__ == "__main__":
    main()