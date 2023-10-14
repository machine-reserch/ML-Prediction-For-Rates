# ML-Prediction-For-Rates: A Comprehensive Guide

## Overview:

The current repository provides a comprehensive framework for predicting reaction rates using machine learning (ML) techniques. It comprises datasets, pre-trained models, and an easy-to-use prediction script for practical applications.

## Datasets:

1. **Complete Datasets**: 
   - These datasets do not contain data division and are saved in files named **FNN_dataset.csv** and **SVR+GPR_dataset.csv**.

2. **Dataset Division**: 
   - We have incorporated two division methodologies: Leave-One-Group-Out (LOGO) and K-Fold. The divisions for these methods are available in the **FNN_loo** and **SVR+GPR_loo** directories. Each directory contains:
     - **DGLOO-testX**: Single reaction data reserved for model testing.
     - **DGLOO-trainX**: Dataset used for model training.
     - **DGLOO-valX**: Validation dataset.
     - **DGLOO-newX**: The dataset after excluding the single reaction under consideration.

3. **Repeated K-Fold Validation Dataset**: 
   - These datasets are found in the **FNN_cross** and **SVR+GPR_cross** directories. Notably, these datasets are dynamically generated during the model evaluation process.

## Pre-Trained Models:

Our framework has undergone rigorous training, and the top 50 models (based on performance metrics) are stored in the **Selecting_model_weight** directory for immediate use.

## Implementation:

To utilize our framework for K-Fold evaluation, users can directly execute the **FNN.py** script.

## Application:

For predicting the reaction rates, we offer an easy-to-use script. Follow these steps:

1. **Molecular Descriptor Calculation**: 
   - Obtain molecular descriptors for the following five attributes: **AATS3s, AATS3m, MATS2dv, GATS1c,** and **SIC3**.

2. **BDE Energy Calculation**: 
   - Determine the Bond Dissociation Energy (BDE) of the reaction you wish to predict using the **M062X/6-311+G(d,p)** level of theory. Remarkably, pinpointing the transition state isn't mandatory.

3. **Data Entry**: 
   - Populate the **example.csv** file with the six derived values.

4. **Prediction**: 
   - Execute the **predict.csv** script. This will output the predicted reaction rates in the temperature range of 700-2000K.

We believe this ML framework will be of immense value to researchers and professionals in the field, providing swift and accurate predictions for reaction rates.




