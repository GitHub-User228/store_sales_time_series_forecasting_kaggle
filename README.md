# Kaggle competition: Store Sales - Time Series Forecasting
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-219ebc?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-778da9?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-778dc9?style=for-the-badge)

Repository which contains code for the Kaggle competition: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/)

# Description

This repository covers the following steps:
- EDA (data cleaning part)
    - dealing with duplicates
    - dealing with missing values (interpolation, time-based imputation)
    - dealing with outliers (IQR filtering, Isolation Forest)
    - rescaling the target feature
- Baseline model
    - LightGBM with default hyperparameters
    - XGBoost with default hyperparameters
- EDA (data analysis part):
    - univariate distribution analysis
    - multivariate distribution analysis with target feature
    - time series analysis
    - correlation analysis
- Feature Engineering:
    - time features (year, month, weekday)
    - rolling and expanding features from target feature
    - lag features
- Models after Feature Engineering:
    - LightGBM with default hyperparameters
    - XGBoost with default hyperparameters


# Kaggle results

RMSLE scores:

- Baseline LightGBM - 0.67209
- Baseline XGBoost - 0.64114
- LightGBM after Feature Engineering - 0.49560
- XGBoost after Feature Engineering - 0.51744
