# ML-Prediction-For-Rates
## The dataset without datadivsion are saved in FNN_dataset.csv and SVR+GPR_dataset.csv.
## The entire dataset division result for LOGO and Kfold method are listed as FNN_loo file and SVR+GPR_loo file.
Noting that the three structure for these two file is same
 - DGLOO-test1  the single reaction data for testing model
 - DGLOO-train1 the training dataset
 - DGLOO-val1   the testing dataset
 - DGLOO-new1  the dataset without the single reaction
Also the Repeated-Kfold valuation dataset are listed in FNN_cross file and  SVR+GPR_cross. And it will dynamically generated during the process.
The selecting 50 models during this process are saved in Selecting_model_weight file
Morever,the reader can directly take K-fold experiment by just run the FNN.py.

For application, you can edit the example.csv and run the predict.py to get the predict result.

