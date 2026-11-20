"""
박막 두께 예측 - 데이터 탐색 및 모델 개발
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
print("=" * 80)
print("데이터 로드 중...")
print("=" * 80)
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"훈련 데이터 크기: {train_df.shape}")
print(f"테스트 데이터 크기: {test_df.shape}")
print(f"\n훈련 데이터 컬럼: {list(train_df.columns[:10])}... (총 {len(train_df.columns)} 개)")

# 데이터 구조 파악
print("\n" + "=" * 80)
print("데이터 구조")
print("=" * 80)
print(train_df.info())
print(f"\n훈련 데이터 통계:")
print(train_df.describe())

# 타겟 변수 분석
target_cols = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
print("\n" + "=" * 80)
print("타겟 변수 분석 (박막 두께)")
print("=" * 80)
print(train_df[target_cols].describe())

# 타겟 변수 간의 상관관계
print("\n타겟 변수 간 상관관계:")
print(train_df[target_cols].corr())

# 반사율 스펙트럼의 범위 확인
spectrum_cols = [col for col in train_df.columns if col not in ['id'] + target_cols]
print(f"\n반사율 스펙트럼 데이터 (파장 개수: {len(spectrum_cols)})")
print(f"반사율 범위: [{train_df[spectrum_cols].min().min():.4f}, {train_df[spectrum_cols].max().max():.4f}]")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 각 레이어 두께 분포
for idx, col in enumerate(target_cols):
    ax = axes[idx // 2, idx % 2]
    ax.hist(train_df[col], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(f'{col} 분포')
    ax.set_xlabel('두께 (nm)')
    ax.set_ylabel('샘플 수')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output_01_target_distribution.png', dpi=100)
print("\n[OK] Target variable distribution saved: output_01_target_distribution.png")
plt.close()

# 2. 샘플별 반사율 스펙트럼 시각화
fig, ax = plt.subplots(figsize=(14, 6))
for i in range(min(10, len(train_df))):
    ax.plot(train_df[spectrum_cols].iloc[i], alpha=0.5, label=f'Sample {i}')
ax.set_title('Reflectance Spectrum by Sample (First 10)')
ax.set_xlabel('Wavelength Index')
ax.set_ylabel('Reflectance')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('output_02_spectrum_samples.png', dpi=100)
print("[OK] Reflectance spectra saved: output_02_spectrum_samples.png")
plt.close()

# 3. 반사율 평균 및 표준편차
spectrum_mean = train_df[spectrum_cols].mean()
spectrum_std = train_df[spectrum_cols].std()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(spectrum_mean, label='Mean', linewidth=2)
ax.fill_between(range(len(spectrum_mean)), 
                 spectrum_mean - spectrum_std, 
                 spectrum_mean + spectrum_std,
                 alpha=0.3, label='+/- 1 Std Dev')
ax.set_title('Reflectance Spectrum - Mean and Standard Deviation')
ax.set_xlabel('Wavelength Index')
ax.set_ylabel('Reflectance')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_03_spectrum_statistics.png', dpi=100)
print("[OK] Spectrum statistics saved: output_03_spectrum_statistics.png")
plt.close()

print("\n" + "=" * 80)
print("Data exploration completed")
print("=" * 80)
