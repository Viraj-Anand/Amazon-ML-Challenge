# Amazon-ML-Challenge

My team's attempt at the Amazon ML Challenge: predict how long an Amazon product is (`PRODUCT_LENGTH`) from the information in its listing. The code is in [train.ipynb](train.ipynb).

## The data

`train.csv` has one row per product, with `PRODUCT_ID`, `TITLE`, `BULLET_POINTS`, `DESCRIPTION`, `PRODUCT_TYPE_ID` and the target `PRODUCT_LENGTH`. `test.csv` has the same columns without the length. The data files are not in this repo.

## What the notebook does

1. Drops `PRODUCT_ID`, `TITLE`, `BULLET_POINTS` and `DESCRIPTION`. My notes from the time say the text columns did not relate directly to the target. This leaves **one input, `PRODUCT_TYPE_ID`**.
2. Keeps only the products whose length is between the 10th and 90th percentile, so the shortest and longest 10% are removed.
3. Splits the rows 80% / 20% for training and validation (`random_state=42`) and scales the input with scikit-learn's `RobustScaler`.
4. Trains a small dense network: layers of 128, 256, 256 and 128 units with ReLU, then one output. Adam optimiser, mean squared error (MSE) loss, mean absolute error (MAE) as a metric, 20 epochs.

The notebook also finds outliers with the interquartile range method and plots them, but the model is trained on the percentile-trimmed data from step 2, not on that cleaned data.

## Results

- After epoch 20, the training loss (MSE) was 920.1 and the validation loss was 880.5. The MAE was about 25.7 on the training rows and 25.6 on the validation rows.
- With the challenge's scoring formula, `max(0, 100 x (1 - MAPE))`, the score is **96.27**. This is measured on the 20% held-out part of the trimmed training data. It is **not** a leaderboard score.
- The last cells predict lengths for `test.csv`, but the notebook does not save a submission file.

## Limitations and what I would change

- With one input, the network mostly learns "the usual length for this product type".
- The 96.27 is measured on data with the shortest and longest products removed, so it would be lower on the full range. I did not compare with a simple baseline, such as always predicting the average length, so I cannot say how much the network adds.
- The notebook fits the scaler again on `test.csv` instead of reusing the one fitted on the training rows. That is a mistake.
- My notes say we first tried random forest, linear regression and XGBoost. That code is not in this repo.
- Next steps: use the text columns (for example TF-IDF of the title and description), keep the full range of lengths, reuse the training scaler, and compare against baselines.

## How to run

```
pip install -r requirements.txt
```

Put `train.csv` and `test.csv` next to the notebook, open `train.ipynb` in Jupyter or VS Code and run it from the top. I used Python 3.9.
