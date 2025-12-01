🏠 Real Estate Price Prediction using Regression Models
🔍 Overview

This project focuses on predicting real estate prices using multiple machine learning regression techniques.
The workflow includes data loading, preprocessing, exploratory analysis, feature engineering, model training, hyperparameter tuning, PCA dimensionality reduction, model comparison, and final evaluation on a test dataset.

The goal is to identify which regression model best predicts housing prices and understand which features most strongly influence the target variable.

📁 Dataset

The dataset consists of training and test subsets containing information about real estate properties such as:

Number of bedrooms

Bathrooms

Living area (m²)

Grade and condition indicators

Renovation status

Presence of basement, lavatory, view quality

Quartile zone

Final price (target variable)

Preprocessing steps included:

Handling missing values

Combining train and test sets for unified EDA

Converting boolean features to numerical (0/1)

Dropping non-informative features (date, month)

Scaling numeric features using StandardScaler

🛠️ Tools & Technologies

Python

Pandas, NumPy (data manipulation)

Matplotlib, Seaborn (visualization)

Scikit-learn

Linear Regression, Ridge Regression

Polynomial Regression

Decision Tree Regressor

SVR (LinearSVR)

PCA

GridSearchCV for hyperparameter tuning

Jupyter Notebook / Google Colab

🔄 Project Steps
1️⃣ Data Loading & Inspection

Loaded training and test datasets

Merged them for unified analysis

Examined structure, types, missing values, and descriptive statistics

2️⃣ Exploratory Data Analysis

Histograms (price distribution)

Correlation heatmap

Pairplots

Boxplots for categorical relationships (e.g., month vs. price)

3️⃣ Data Cleaning & Feature Engineering

Converted boolean features to numerical

Removed columns with low predictive value

Scaled numeric features

Split train/validation sets (90/10 split)

4️⃣ Model Training

Trained and evaluated:

Linear Regression

Ridge Regression (GridSearch over alpha)

Polynomial Regression (hyperparameter: degree 1–4)

Decision Tree Regressor (GridSearch over max_depth, min_samples_split)

SVR (LinearSVR) (GridSearch over C and epsilon)

5️⃣ PCA Analysis

Performed PCA (90% explained variance)

Re-trained Polynomial Regression, Decision Tree, and SVR on PCA-transformed data

Compared original vs. PCA performance

6️⃣ Model Comparison

Evaluated models using:

RMSE

R² Score

MAE

Results were visualized with bar charts and prediction vs actual scatter plots.

7️⃣ Final Model Selection & Testing

The top-performing models were retrained on the full training dataset and tested on the provided test set.
Final results include comparison of predicted vs. true prices for:

Polynomial Regression

Decision Tree

SVR

📈 Results

Insights from model performance include:

Polynomial Regression showed the strongest predictive accuracy.

Decision Trees capture non-linear relationships but may overfit.

SVR performed well but was sensitive to hyperparameter tuning.

PCA reduced dimensionality but slightly decreased performance in all models.

Models performed better in the original feature space than after PCA transformation.

Comparisons of test set RMSE, MAE, and R² were visualized for clarity.
