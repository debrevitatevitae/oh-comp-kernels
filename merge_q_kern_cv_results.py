import pandas as pd
import os

from project_directories import RESULTS_DIR
import glob

if __name__ == '__main__':
    open_files = glob.glob(os.path.join(
        RESULTS_DIR, "q_kern_accuracy_cv_*.csv"))

    # Initialize an empty DataFrame
    merged_df = pd.DataFrame()

    # Loop through the list of open files
    for file in open_files:
        # Check if the file exists
        if os.path.isfile(file):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)

            # Extract parts of the filename
            parts = file.split('_')
            df['embedding_name'] = parts[-1].split('.')[0]

            # Append the DataFrame to the merged DataFrame
            merged_df = pd.concat([merged_df, df])
        else:
            print(f"File {file} does not exist.")

    # Write the merged DataFrame to a new CSV file
    merged_df.to_csv(RESULTS_DIR / "q_kern_accuracy_cv.csv", index=False)
