"""
Final Submission - Using full training data with optimized hyperparameters
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Final Model - Training on full dataset with optimized parameters")
print("=" * 80)

# Load all data
print("\nLoading full dataset...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

target_cols = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
spectrum_cols = [col for col in train_df.columns if col not in ['id'] + target_cols]

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Use ALL training data (but with memory-efficient approach)
X_train_full = train_df[spectrum_cols].values
y_train_full = train_df[target_cols].values
X_test = test_df[spectrum_cols].values

# Standardization
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
print("Applying PCA...")
pca = PCA(n_components=120, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
explained_var = pca.explained_variance_ratio_.sum()
print(f"PCA - Explained variance: {explained_var:.4f}")
print(f"Reduced dimensions: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]}")

best_models = {}
predictions_test = {}
metrics = {}

print("\n" + "=" * 80)
print("Training final models for each layer (using ALL data)...")
print("=" * 80)

for layer_idx, layer_name in enumerate(target_cols):
    print(f"\n--- {layer_name} ---")
    
    y_layer = y_train_full[:, layer_idx]
    
    # Try multiple regularization strengths and choose best based on initial validation
    # (using first 100k samples for quick validation)
    validation_size = min(100000, len(X_train_pca))
    X_val_check = X_train_pca[:validation_size]
    y_val_check = y_layer[:validation_size]
    
    best_alpha = None
    best_rmse = float('inf')
    
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_val_check, y_val_check)
        y_pred = ridge.predict(X_val_check)
        rmse = np.sqrt(mean_squared_error(y_val_check, y_pred))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
        
        print(f"  Ridge(alpha={alpha:6.1f}) - RMSE: {rmse:.4f}")
    
    print(f"  BEST alpha: {best_alpha} (RMSE: {best_rmse:.4f})")
    
    # Train final model on FULL data
    print(f"  Training final Ridge(alpha={best_alpha}) on ALL {len(X_train_pca)} samples...")
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train_pca, y_layer)
    
    # Predictions
    y_pred_train = final_model.predict(X_train_pca)
    y_pred_test = final_model.predict(X_test_pca)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_layer, y_pred_train))
    train_mae = mean_absolute_error(y_layer, y_pred_train)
    train_r2 = r2_score(y_layer, y_pred_train)
    print(f"  Final training metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
    
    best_models[layer_name] = final_model
    predictions_test[layer_name] = y_pred_test
    metrics[layer_name] = {
        'alpha': best_alpha,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
    }

# Create final submission
print("\n" + "=" * 80)
print("Creating final submission file...")
print("=" * 80)

submission = pd.DataFrame()
submission['id'] = test_df['id']

for layer_name in target_cols:
    submission[layer_name] = predictions_test[layer_name]

submission.to_csv('submission_final.csv', index=False)
print(f"Final submission created: submission_final.csv")
print(f"Shape: {submission.shape}\n")
print(submission.head(10))

# Save models
print("\n" + "=" * 80)
print("Saving final models...")
print("=" * 80)

with open('models_final.pkl', 'wb') as f:
    pickle.dump({
        'models': best_models,
        'scaler': scaler,
        'pca': pca
    }, f)

print("Models saved: models_final.pkl")

# Final summary
print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 80)

total_rmse = 0
total_mae = 0
total_r2 = 0

for layer_name in target_cols:
    m = metrics[layer_name]
    print(f"\n{layer_name}:")
    print(f"  Hyperparameter: Ridge(alpha={m['alpha']})")
    print(f"  Training RMSE: {m['train_rmse']:.4f} nm")
    print(f"  Training MAE: {m['train_mae']:.4f} nm")
    print(f"  Training R2: {m['train_r2']:.4f}")
    
    total_rmse += m['train_rmse']
    total_mae += m['train_mae']
    total_r2 += m['train_r2']

print(f"\nAverage across all layers:")
print(f"  Average RMSE: {total_rmse / 4:.4f} nm")
print(f"  Average MAE: {total_mae / 4:.4f} nm")
print(f"  Average R2: {total_r2 / 4:.4f}")

print("\n" + "=" * 80)
print("Final model training completed successfully!")
print("=" * 80)
