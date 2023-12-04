# Predicting open hole composite failure using classical- and quantum-kernel SVM

## Preliminaries
### Get the raw data
Data is available at the following link: https://zenodo.org/records/7409612.

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

## Description of the files
- `rbf_kta.py`: studies the concentration around the average of the RBF KTA for different batch sizes.
- `pp_rbf_kta.py`: plots histograms for each batch size with the values of the KTA compared to the full-training set KTA.
- `rbf_kta_opt.py`: maximizes KTA of the RBF kernel.
- `pp_rbf_kta_opt.py`: plots the KTA vs epochs for the RBF KTA maximization.
- `rbf_accuracy_cv.py`: runs a grid-search cross validataton for different $(C,\,\gamma)$ combinations of the RBF SVM, including $\gamma_{\mathrm{opt}}$.
- `rbf_accuracy_test.py`: computes the test set accuracy for the $(C,\,\gamma)_{\mathrm{opt}}$, which is the hyperparameters combination with the highest validation set accuracy as found from `rbf_accuracy_cv.py`.
