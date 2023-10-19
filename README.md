# About The Project
This project contains a hate speech dataset that supports sequence and token classification. It also contains the scripts to train a ConvBERT and Distil RoBERTa model, and evaluate the results. You can access the full dataset in *data\initial_data*.

# Usage
Here are the files and their uses:
* **commons.py** holds the constants and classes.
* **read_data.py** holds the functions to read the data from the data folder.
* **write_data.py** holds the functions to calculate majority vote and write the data to the output folder.
* **tokenize_data.py** holds the functions to tokenize the data and write the tokenized data to the output folder.
* **print_statistics.py** holds the functions to print the statistics of the data.
* **trainer.py** holds the functions to train the models.
* **result_report.py** holds the functions to print the results of the training process.
* **error_analysis.py** holds the functions to analyze the errors of the models.

You can run the files in the following order:
1. **read_data.py**
2. **write_data.py**
3. **tokenize_data.py**
4. **print_statistics.py**
5. **trainer.py**
6. **result_report.py**
7. **error_analysis.py**

Also make sure that you run **tokenize_data.py** if you change the model in **commons.py**.

# TODO
It is not efficient to load both of the models at the same time, but that is what is being done right now. It would be better to only load the model that is being used at the moment. However, it doesn't seem to be a problem for the current data set, and my GPU. I will fix this.

It is also not efficient to load the entire data to GPU at once. It would be better to load the data in batches. However, it doesn't seem to be a problem for the current data set, and my GPU. I will fix this.

