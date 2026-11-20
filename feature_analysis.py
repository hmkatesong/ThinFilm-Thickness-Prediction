"""
Best Features Analysis - Identify which wavelengths are most predictive
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Feature Importance Analysis")
print("=" * 80)

train_df = pd.read_csv('data/train.csv')

target_cols = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
spectrum_cols = [col for col in train_df.columns if col not in ['id'] + target_cols]

# Sample
np.random.seed(42)
sample_size = 50000
sample_indices = np.random.choice(len(train_df), sample_size, replace=False)

X_train = train_df[spectrum_cols].iloc[sample_indices].values
y_train = train_df[target_cols].iloc[sample_indices].values

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Get feature importance for first layer using Ridge coefficients
print("\nAnalyzing feature importance for layer_1...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train[:, 0])

coefficients = np.abs(ridge.coef_)
top_features_idx = np.argsort(coefficients)[-20:][::-1]
top_features = [spectrum_cols[i] for i in top_features_idx]

print(f"Top 20 most important wavelengths (by Ridge coefficients):")
for i, feat_idx in enumerate(top_features_idx):
    print(f"  {i+1}. Wavelength {spectrum_cols[feat_idx]}: coef = {ridge.coef_[feat_idx]:.6f}")

# Try model with top features only
print("\nTraining model with top 50 features only...")
top_50_idx = np.argsort(coefficients)[-50:]

X_train_top = X_train_scaled[:, top_50_idx]

ridge_top = Ridge(alpha=10.0)
ridge_top.fit(X_train_top[:40000], y_train[:40000, 0])

y_pred_full = ridge_top.predict(X_train_top)
rmse_full = np.sqrt(np.mean((y_pred_full - y_train[:, 0])**2))
print(f"Ridge with top 50 features - Training RMSE: {rmse_full:.4f} nm")

print("\n" + "=" * 80)
print("Analysis complete")
print("=" * 80)
