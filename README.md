# 🏠 House Price Prediction Model

A machine learning project that predicts house prices using property features such as size, bedrooms, location, and more. Two regression models are trained and compared — Linear Regression and Gradient Boosting.

---

## 📋 Project Overview

This project builds a House Price Prediction system using real estate data. The system performs complete data preprocessing, feature engineering, model training, evaluation, and visualization. Two models are trained and compared to find the best performing one.

The dataset used is the House Price Prediction Dataset from Kaggle, containing features like square footage, number of bedrooms, bathrooms, location, and more.

---

## 🎯 Objectives

- Perform preprocessing on features like square footage, number of bedrooms, and location
- Train regression models (Linear Regression and Gradient Boosting)
- Visualize predicted prices compared to actual prices
- Evaluate models using Mean Absolute Error (MAE) and RMSE

---

## 🗂️ Project Files

| File | Description |
|------|-------------|
| `House_price.ipynb` | Jupyter Notebook with step-by-step implementation and outputs |
| `House_price.py` | Clean Python script with full comments |
| `House data.csv` | Dataset used for training and testing |

---

## ✨ Features

- **Data Exploration** — Dataset info, missing values check, column preview
- **Outlier Detection** — Boxplots and IQR method to identify outliers
- **Outlier Handling** — Remove impossible values and cap extreme values using Winsorization
- **Feature Engineering** — Created `house_age` and `is_renovated` from existing columns
- **Target Encoding** — Encoded `city` and `statezip` using mean price (train data only)
- **Feature Scaling** — StandardScaler applied on train data only to prevent data leakage
- **Two Models Trained** — Linear Regression and Gradient Boosting
- **Model Evaluation** — R² Score, MAE and RMSE for both models
- **Visualization** — Actual vs Predicted price scatter plots for both models

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Primary programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Scikit-learn | Machine learning models and evaluation |

---

## 📦 Installation & Setup

### Step 1: Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 2: Download Dataset
Download the House Price Prediction Dataset from Kaggle and save it as `House data.csv` in the same folder as the project files.

### Step 3: Run the Project
```bash
python House_price.py
```
Or open `House_price.ipynb` in Jupyter Notebook and run cells one by one.

---

## 🔄 Project Pipeline

```
Load Dataset
     ↓
Data Exploration (info, missing values, preview)
     ↓
Drop Irrelevant Columns (date, street, country)
     ↓
Outlier Detection (Boxplots + IQR)
     ↓
Outlier Handling (Remove impossible values + Winsorization)
     ↓
Remove Invalid Rows (price = 0)
     ↓
Feature Engineering (house_age, is_renovated)
     ↓
Train/Test Split (80% train, 20% test)
     ↓
Target Encoding (city, statezip)
     ↓
Feature Scaling (StandardScaler)
     ↓
Train Models (Linear Regression + Gradient Boosting)
     ↓
Evaluate Models (R², MAE, RMSE)
     ↓
Visualize Results (Actual vs Predicted)
```

---

## 🧹 Data Preprocessing

### Columns Dropped
| Column | Reason |
|--------|--------|
| `date` | Not useful for prediction |
| `street` | Too unique, causes overfitting |
| `country` | Only one value (USA) |

### Outlier Handling
- **Bedrooms** — Removed rows with 0 or more than 7 bedrooms (impossible values)
- **Bathrooms** — Removed rows with less than 0.5 or more than 6 bathrooms
- **Price, sqft columns** — Capped using IQR Winsorization
- **Size columns** — Lower bound set to 0 (negative sqft not possible)
- **Price = 0** — Removed rows with zero price

### Feature Engineering
```python
# Age of house
dataset['house_age'] = 2024 - dataset['yr_built']

# Was it renovated?
dataset['is_renovated'] = (dataset['yr_renovated'] > 0).astype(int)
```

### Target Encoding
`city` and `statezip` columns were encoded using the mean house price of each category calculated from **training data only** to prevent data leakage.

### Feature Scaling
StandardScaler was fitted on training data only and applied to both train and test sets.

---

## 🤖 Models

### Linear Regression
```python
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
```

### Gradient Boosting
```python
gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
```

---

## 📊 Model Results

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | ~80% | ~$80,621 | ~$113,094 |
| Gradient Boosting | ~83% | ~$68,062 | ~$102,687 |

**Gradient Boosting outperforms Linear Regression on all metrics!**

---

## 📈 Visualization

Two scatter plots are generated comparing Actual vs Predicted prices:

- **Blue plot** — Linear Regression predictions
- **Green plot** — Gradient Boosting predictions
- **Red dashed line** — Perfect prediction line (actual = predicted)

Points closer to the red line indicate more accurate predictions.

---

## 💡 Key Insights

- Gradient Boosting performs better than Linear Regression for this dataset
- House price is not linear — Gradient Boosting handles non-linearity better
- Location (`city`, `statezip`) has strong influence on price — Target Encoding captures this well
- Feature engineering (`house_age`, `is_renovated`) improved model understanding
- Both models struggle slightly with very high-priced homes due to limited data

---

## ⚠️ Important Notes

- Always fit the scaler on training data only — fitting on full data causes data leakage
- Target encoding must also be calculated from training data only
- Winsorization preserves data rows while handling extreme values

---

## 👤 Author

**Muhammad Umar**
AI/ML Engineering Intern | Python Developer
University of Agriculture Faisalabad
LinkedIn: [linkedin.com/in/umar-ml](https://linkedin.com/in/umar-ml)

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
