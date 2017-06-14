HOW TO RUN THE NOTEBOOKS:

There are three Python3 Notebooks and 1 R script:
- DataExploration.ipynb deals with some insights about the data and takes care of some cleaning in the data format
- timeSeriesAnalysis.r is an R script that reads the train and test files and tries to extract come time-series relevant informations by fitting an ARIMA process
- Transform features.ipynb takes the clean train and test files and adds all the engineered features we came up with
- Feature_model_selection_and_final_prediction.ipynb reads the train and test files with all the engineered features and generates a prediction file (after carrying out a feature selection and model selection process)


Instructions:

1) Put the train.csv and test.csv in the folder data/
2) Run the notebook 'DataExploration.ipynb' (or at least the first few cells, in order to create train_clean.csv and test_clean.csv)
3) Run the R script 'timeSeriesAnalysis.r' so that it generates in the folder data/other the files train_orders.csv and test_orders.csv containing the time series related features.
4) Run the 'Transform features.ipynb' notebook and generate the files data/train_predictedAges.csv and data/test_all_features.csv
5) Run the 'Feature_model_selection_and_final_prediction.ipynb' to create the predictions.csv and predictions_kaggle_format.csv files.


Note (1): We have already provided you the time series analysis files in data/other/* so that you can skip the step 3 and only have to run the notebooks.

Note (2): The notebooks (in particular the last one) can take a lot of time to execute. We have reduced the computation time as much as possible, but it can still be expensive.

Note (3): We provide you two predictions files: the first one is named 'predictions.csv' and contains all the informations of the test file with the target column DEFAULT PAYMENT JAN filled with our predictions. The second file is named 'predictions_kaggle_format.csv' and contains the same predictions but with the usual Kaggle format, with two columns: (CUST_COD, DEFAULT PAYMENT JAN)