import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from project_directories import RESULTS_DIR

from utils import load_split_data


if __name__ == '__main__':
    start = time.time()

    np.random.seed(42)

    # data loading, splitting and scaling
    X_train, _, y_train, _ = load_split_data(test_size=0.2)
    N = len(y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Read the optimal gamma value found during KTA optimization
    df_kta_opt = pd.read_csv(RESULTS_DIR / "rbf_kta_opt.csv")
    # index 3 corresponds to 15 epochs, when the KTA stops growing monotonically
    gamma_opt = df_kta_opt["gamma"][3]

    # Define the parameter grid for GridSearchCV
    param_grid = {'C': np.logspace(-1, 2, 4),
                  'gamma': [0.01, 0.1, gamma_opt, 10., 100.]}

    # create SVC
    svc = SVC(kernel="rbf")

    # create a GridSearchCV and fit to the data
    grid_search = GridSearchCV(
        estimator=svc, param_grid=param_grid, scoring='accuracy', cv=10)
    grid_search.fit(X_train_scaled, y_train)

    # store CV results in a DataFrame
    df_results = pd.DataFrame(grid_search.cv_results_)

    # Extract the mean and standaed deviation of the validation error and save to csv file
    selected_columns = ['param_C', 'param_gamma',
                        'mean_test_score', 'std_test_score']
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    df_results[selected_columns].to_csv(
        RESULTS_DIR / f'{python_file_name_no_ext}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")