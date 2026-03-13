### Train, Validation, and Test set Split for Time-Series Dataset with Text Descriptions

After generating textual descriptions for the time-series data, we split the output files into training, validation, and test sets. This ensures the datasets are ready for model training and evaluation. It splits the data into **train**, **validation**, and **test** sets according to a predefined ratio (default: 80% train, 10% val, 10% test).

The splitting process is implemented in the following script:  [Dataset Split Code](https://github.com/chang-xu/TimeGen/blob/main/process/dataset_split.py)

### Example Command
```bash
python dataset_split.py --input_dir ./output_files --output_dir ./split_files
```
