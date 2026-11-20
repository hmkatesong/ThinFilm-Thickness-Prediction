"""
Thin Film Thickness Prediction - Optimized Model Training
Using stratified sampling for faster training on large dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Loading data...")
print("=" * 80)

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Extract features and targets
target_cols = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
spectrum_cols = [col for col in train_df.columns if col not in ['id'] + target_cols]

print(f"Original training set size: {len(train_df)}")

# Sample data for faster training (but representative)
sample_size = min(50000, len(train_df))  # Use up to 50k samples
print(f"Using stratified sample: {sample_size} samples")

np.random.seed(42)
sample_indices = np.random.choice(len(train_df), sample_size, replace=False)

X_train = train_df[spectrum_cols].iloc[sample_indices].values
y_train = train_df[target_cols].iloc[sample_indices].values
X_test = test_df[spectrum_cols].values

print(f"Training features shape: {X_train.shape}")
print(f"Training targets shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Standardize features
print("\n" + "=" * 80)
print("Feature Standardization...")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized")

# Train individual models for each layer
print("\n" + "=" * 80)
print("Training models for each layer...")
print("=" * 80)

best_models = {}
predictions_test = {}
metrics = {}

for layer_idx, layer_name in enumerate(target_cols):
    print(f"\n--- {layer_name} ---")
    
    y_layer = y_train[:, layer_idx]
    
    # Simple train-test split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_layer, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"  Training Ridge regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    
    y_pred_val = ridge.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    print(f"    Ridge - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    print(f"  Training Lasso regression...")
    lasso = Lasso(alpha=0.01, max_iter=1000)
    lasso.fit(X_tr, y_tr)
    
    y_pred_val_lasso = lasso.predict(X_val)
    rmse_lasso = np.sqrt(mean_squared_error(y_val, y_pred_val_lasso))
    mae_lasso = mean_absolute_error(y_val, y_pred_val_lasso)
    r2_lasso = r2_score(y_val, y_pred_val_lasso)
    print(f"    Lasso - RMSE: {rmse_lasso:.4f}, MAE: {mae_lasso:.4f}, R2: {r2_lasso:.4f}")
    
    # Choose best model
    if rmse < rmse_lasso:
        best_model = ridge
        best_model_name = "Ridge"
        best_metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
    else:
        best_model = lasso
        best_model_name = "Lasso"
        best_metrics = {'rmse': rmse_lasso, 'mae': mae_lasso, 'r2': r2_lasso}
    
    print(f"  BEST: {best_model_name}")
    
    # Retrain on full training set for final model
    final_model = Ridge(alpha=1.0) if best_model_name == "Ridge" else Lasso(alpha=0.01, max_iter=1000)
    final_model.fit(X_train_scaled, y_layer)
    
    # Predictions
    y_pred_train = final_model.predict(X_train_scaled)
    y_pred_test = final_model.predict(X_test_scaled)
    
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_layer, y_pred_train))
    train_mae = mean_absolute_error(y_layer, y_pred_train)
    train_r2 = r2_score(y_layer, y_pred_train)
    print(f"  Training metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
    
    best_models[layer_name] = final_model
    predictions_test[layer_name] = y_pred_test
    metrics[layer_name] = {
        'model': best_model_name,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': best_metrics['rmse']
    }

# Create submission
print("\n" + "=" * 80)
print("Creating submission file...")
print("=" * 80)

submission = pd.DataFrame()
submission['id'] = test_df['id']

for layer_name in target_cols:
    submission[layer_name] = predictions_test[layer_name]

submission.to_csv('submission.csv', index=False)
print(f"Submission file created: submission.csv")
print(f"Shape: {submission.shape}")
print(f"\nFirst 10 rows:")
print(submission.head(10))

# Save models and scaler
print("\n" + "=" * 80)
print("Saving models...")
print("=" * 80)

with open('models.pkl', 'wb') as f:
    pickle.dump(best_models, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models and scaler saved")

# Summary report
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

for layer_name in target_cols:
    m = metrics[layer_name]
    print(f"\n{layer_name}:")
    print(f"  Model: {m['model']}")
    print(f"  Training RMSE: {m['train_rmse']:.4f} nm")
    print(f"  Training MAE: {m['train_mae']:.4f} nm")
    print(f"  Training R2: {m['train_r2']:.4f}")
    print(f"  Validation RMSE: {m['val_rmse']:.4f} nm")

print("\n" + "=" * 80)
print("Model training completed successfully!")
print("=" * 80)
