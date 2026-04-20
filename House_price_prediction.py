# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
dataset = pd.read_csv("House data.csv")

# Display column names
dataset.columns

# Display dataset info (datatypes, non-null counts)
dataset.info()

# Check for missing values
dataset.isnull().sum()

# Preview first 3 rows
dataset.head(3)

# Drop irrelevant columns
dataset = dataset.drop(columns=['date', 'street', 'country'])
dataset.head(3)

# Key columns to check for outliers
cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 
        'sqft_lot', 'sqft_above', 'sqft_basement']

# Boxplots to visually see outliers
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(cols):
    axes[i].boxplot(dataset[col].dropna())
    axes[i].set_title(col)

plt.tight_layout()
plt.show()

# IQR method to count outliers
for col in cols:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = dataset[(dataset[col] < Q1 - 1.5 * IQR) | 
                       (dataset[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

# Step 1: Remove impossible values (bedrooms & bathrooms)
dataset = dataset[dataset['bedrooms'] <= 7]
dataset = dataset[dataset['bedrooms'] >= 1]
dataset = dataset[dataset['bathrooms'] <= 6]
dataset = dataset[dataset['bathrooms'] >= 0.5]

print(f"After removing impossible values: {dataset.shape}")

# Step 2: Cap outliers using IQR (Winsorization) for remaining columns
cols_to_cap = ['price', 'sqft_living', 'sqft_lot', 
               'sqft_above', 'sqft_basement']

for col in cols_to_cap:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    dataset[col] = dataset[col].clip(lower=lower, upper=upper)
    print(f"{col} → capped at [{lower:.0f}, {upper:.0f}]")

print(f"\nFinal shape: {dataset.shape}")
print(dataset[['price', 'bedrooms', 'bathrooms', 
               'sqft_living', 'sqft_lot']].describe())

# Step 3: Fix negative lower bounds for size columns
cols_to_cap = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

for col in cols_to_cap:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = max(0, Q1 - 1.5 * IQR)  # Never allow negative lower bound
    upper = Q3 + 1.5 * IQR
    dataset[col] = dataset[col].clip(lower=lower, upper=upper)
    print(f"{col} → capped at [{lower:.0f}, {upper:.0f}]")

print(dataset[['sqft_living', 'sqft_lot', 
               'sqft_above', 'sqft_basement']].min())

# Remove rows where price is 0 (invalid data)
dataset = dataset[dataset['price'] > 0]
print(f"Shape after removing 0-price rows: {dataset.shape}")

# Feature Engineering: Age of house
dataset['house_age'] = 2024 - dataset['yr_built']

# Feature Engineering: Was it renovated?
dataset['is_renovated'] = (dataset['yr_renovated'] > 0).astype(int)

# Drop original year columns (now replaced by engineered features)
dataset = dataset.drop(columns=['yr_built', 'yr_renovated'])

# Separate features and target variable
x = dataset.drop("price", axis=1)
y = dataset["price"]

# Train/Test split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Combine x_train and y_train temporarily for target encoding
train_temp = pd.concat([x_train, y_train], axis=1)

# Target Encoding using train data only (prevents data leakage)
city_mean = train_temp.groupby('city')['price'].mean()
statezip_mean = train_temp.groupby('statezip')['price'].mean()

# Apply encoding and handle unseen values with mean fallback
x_train['city'] = x_train['city'].map(city_mean)
x_test['city'] = x_test['city'].map(city_mean).fillna(city_mean.mean())

x_train['statezip'] = x_train['statezip'].map(statezip_mean)
x_test['statezip'] = x_test['statezip'].map(statezip_mean).fillna(
    statezip_mean.mean())

# Feature Scaling using StandardScaler (fit on train only)
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Evaluate Linear Regression Model (R² Score)
print("Linear Regression R² Score:", lr_model.score(x_test, y_test) * 100)

# Train Gradient Boosting Model with tuned hyperparameters
gb_model = GradientBoostingRegressor(
    n_estimators=500,       # Number of trees
    learning_rate=0.05,     # Shrinkage rate
    max_depth=5,            # Tree depth
    min_samples_split=10,   # Minimum samples to split
    random_state=42
)
gb_model.fit(x_train, y_train)

# Evaluate Gradient Boosting Model (R² Score)
print("Gradient Boosting R² Score:", gb_model.score(x_test, y_test) * 100)

# Generate predictions for both models
y_pred_lr = lr_model.predict(x_test)
y_pred_gb = gb_model.predict(x_test)

# Calculate MAE and RMSE for both models
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print("Linear Regression  → MAE:", round(mae_lr, 2), 
      "| RMSE:", round(rmse_lr, 2))
print("Gradient Boosting  → MAE:", round(mae_gb, 2), 
      "| RMSE:", round(rmse_gb, 2))

# Visualize Actual vs Predicted Prices for both models
plt.figure(figsize=(14, 6))

# Linear Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted")

# Gradient Boosting Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_gb, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Gradient Boosting: Actual vs Predicted")

plt.tight_layout()
plt.show()