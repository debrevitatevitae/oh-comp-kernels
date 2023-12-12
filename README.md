# Predicting open hole composite failure using classical- and quantum-kernel SVM

## Preliminaries
### Get the raw data
Data is publicly available at the following link: https://zenodo.org/records/7409612.

Download in the `data/raw/` directory
```
pip3 install zenodo_get
cd data/raw
zenodo_get 10.5281/zenodo.7409612
```

### Label the data
Simply run the file `label_data.py`:
```
make run FILE=label_data.py
```

## General logic of the code structure
Except for 'general-purpose' files (`label_data.py`, `project_directories.py` and `utils.py`), scripts without the `pp_` prefix produce CSV files that can be later loaded by `pp_` (post-processing) scripts for graphics.

## Description of the files
- `rbf_kta.py`: studies the concentration around the average of the RBF KTA for different batch sizes.
- `pp_rbf_kta.py`: plots histograms for each batch size with the values of the KTA compared to the full-training set KTA.
- `rbf_kta_opt.py`: maximizes KTA of the RBF kernel.
- `pp_rbf_kta_opt.py`: plots the KTA vs epochs for the RBF KTA maximization.
- `rbf_accuracy_cv.py`: runs a grid-search cross validataton for different $(C,\,\gamma)$ combinations of the RBF SVM, including $\gamma_{\mathrm{opt}}$.
- `pp_rbf_accuracy_cv.py`: plots a heatmap of the validation accuracy for different $(C,\,\gamma)$ combinations of the RBF SVM.
- `rbf_accuracy_test.py`: computes the test set accuracy after training for RBF. The training set size is allowed to increase. The hyperparameters are fixed to $(C,\,\gamma)_{\mathrm{opt}}$, corresponding to the highest KTA (see `rbf_accuracy_cv.py`).
- `pp_rbf_accuracy_test.py`: produces an error-bar plot of the test accuracy of the RBF SVM with $(C,\,\gamma)_{\mathrm{opt}}$ for increasing size of the training set.
- `q_kern_select.py`: explores the KTA of a grid of quantum kernels (Hardware Efficient Ansatz embedding). Width and depth are varied. Saves statistics to a csv and reports the maximum KTA and the variance achieved by every model.
- `pp_q_kern_select.py`: produces heatmaps of the maximum KTA and the KTA variance for all the width/depth combinations of the Hardware Efficient Ansatz (HEA) kernels considered in `q_kern_select.py`. These charts are used for kernel selection. 
- `q_kern_kta_opt`: kta optimization of a quantum kernel identified by a width and a depth of the embedding. The optimization is repeated for multiple initializations of the variational parameters.
- `pp_q_kern_kta_opt`: plots the optimization histories of the best run for several different quantum kernels.