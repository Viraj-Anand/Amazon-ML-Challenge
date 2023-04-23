# Amazon-ML-Challenge



## Project Overview

In this project, we aimed to build a machine learning model to predict the target variable. Initially, we started with a dataset that contained several features including PRODUCT_ID, TITLE, BULLET_POINTS, DESCRIPTION, etc. However, we found that these features were not directly related to the target variable, and they were showing high correlation with PRODUCT_ID_TYPE. Therefore, we removed these features from the dataset.

Next, we detected and removed outliers from the remaining data using the interquantile method. We used "Robust Scalar" to scale the data as it is not very much affected by outliers.

We experimented with various machine learning algorithms including RANDOM FOREST, LINEAR regressor and XGBoost, but the model's performance was not up to the mark, and we couldn't minimize the loss very much.

Finally, we decided to build a neural network with a layering of (128X256X256X128) and fit the data into it. We trained the model for 20 epochs using the ADAM optimizer and used mean absolute error for calculating the loss.

After 20 epochs, the training loss was 920.1493, and the validation loss was 880.455. Using the scoring metric given in the problem statement, we got an accuracy of 96.2% on the test data.

## Conclusion

In conclusion, our neural network model with a layering of (128X256X256X128) performed well and achieved an accuracy of 96.2% on the test data. We hope this README file provides you with the necessary information regarding our approach and methodology for this project.
