# Predicting open hole composite failure using classical- and quantum-kernel SVM

## Usage
### Get the raw data
Data is available at the following link: https://zenodo.org/records/7409612.

Download in the `data/raw/` directory
```pip3 install zenodo_get
cd data/raw
zenodo_get 10.5281/zenodo.7409612
```

### Label the data
Simply run the file `label_data.py`:
```make run FILE=label_data.py```
