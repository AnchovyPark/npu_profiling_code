import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import glob

# matmul 폴더 내 모든 CSV 파일 찾기
csv_files = glob.glob('*.csv')
print(f"Found CSV files: {csv_files}")
print("=" * 80)

# 모든 데이터를 담을 리스트
all_data = []

# 각 CSV 파일 읽기
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        # M, K, N, latency_per_cycle 컬럼이 있는지 확인
        required_cols = ['M', 'K', 'N', 'latency_per_cycle']
        if all(col in df.columns for col in required_cols):
            df['source_file'] = csv_file
            all_data.append(df[required_cols + ['source_file']])
            print(f"[OK] {csv_file}: {len(df)} rows")
        else:
            missing = [col for col in required_cols if col not in df.columns]
            print(f"[SKIP] {csv_file}: missing columns {missing}")
    except Exception as e:
        print(f"[ERROR] {csv_file}: {e}")

print("=" * 80)

# 모든 데이터 합치기
if not all_data:
    print("No data available for analysis.")
    exit(1)

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal data points: {len(combined_df)}")
print(f"M range: {combined_df['M'].min()} ~ {combined_df['M'].max()}")
print(f"K range: {combined_df['K'].min()} ~ {combined_df['K'].max()}")
print(f"N range: {combined_df['N'].min()} ~ {combined_df['N'].max()}")
print(f"latency_per_cycle range: {combined_df['latency_per_cycle'].min():.8f} ~ {combined_df['latency_per_cycle'].max():.8f}")

# 종속 변수
y = combined_df['latency_per_cycle'].values

# ============================================================================
# Model 1: Basic - M, K, N only
# ============================================================================
print("\n" + "=" * 80)
print("Model 1: Basic Linear Regression")
print("=" * 80)
print("Formula: latency_per_cycle = a + b1*M + b2*K + b3*N")

X1 = combined_df[['M', 'K', 'N']].values
model1 = LinearRegression()
model1.fit(X1, y)
y_pred1 = model1.predict(X1)

print(f"\nCoefficients:")
print(f"  a (intercept) = {model1.intercept_:.10e}")
print(f"  b1 (M) = {model1.coef_[0]:.10e}")
print(f"  b2 (K) = {model1.coef_[1]:.10e}")
print(f"  b3 (N) = {model1.coef_[2]:.10e}")
print(f"\nR² = {r2_score(y, y_pred1):.6f}")
print(f"MAE = {mean_absolute_error(y, y_pred1):.10e}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_pred1)):.10e}")

# ============================================================================
# Model 2: With M×K and K×N interaction terms
# ============================================================================
print("\n" + "=" * 80)
print("Model 2: Linear Regression with M*K and K*N interactions")
print("=" * 80)
print("Formula: latency_per_cycle = a + b1*M + b2*K + b3*N + b4*(M*K) + b5*(K*N)")

X2 = np.column_stack([
    combined_df['M'].values,
    combined_df['K'].values,
    combined_df['N'].values,
    combined_df['M'].values * combined_df['K'].values,  # M*K
    combined_df['K'].values * combined_df['N'].values   # K*N
])

model2 = LinearRegression()
model2.fit(X2, y)
y_pred2 = model2.predict(X2)

print(f"\nCoefficients:")
print(f"  a (intercept) = {model2.intercept_:.10e}")
print(f"  b1 (M) = {model2.coef_[0]:.10e}")
print(f"  b2 (K) = {model2.coef_[1]:.10e}")
print(f"  b3 (N) = {model2.coef_[2]:.10e}")
print(f"  b4 (M*K) = {model2.coef_[3]:.10e}")
print(f"  b5 (K*N) = {model2.coef_[4]:.10e}")
print(f"\nR² = {r2_score(y, y_pred2):.6f}")
print(f"MAE = {mean_absolute_error(y, y_pred2):.10e}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_pred2)):.10e}")

# ============================================================================
# Model 3: With all interaction terms (M×K, M×N, K×N)
# ============================================================================
print("\n" + "=" * 80)
print("Model 3: Linear Regression with all interaction terms")
print("=" * 80)
print("Formula: latency_per_cycle = a + b1*M + b2*K + b3*N + b4*(M*K) + b5*(M*N) + b6*(K*N)")

X3 = np.column_stack([
    combined_df['M'].values,
    combined_df['K'].values,
    combined_df['N'].values,
    combined_df['M'].values * combined_df['K'].values,  # M*K
    combined_df['M'].values * combined_df['N'].values,  # M*N
    combined_df['K'].values * combined_df['N'].values   # K*N
])

model3 = LinearRegression()
model3.fit(X3, y)
y_pred3 = model3.predict(X3)

print(f"\nCoefficients:")
print(f"  a (intercept) = {model3.intercept_:.10e}")
print(f"  b1 (M) = {model3.coef_[0]:.10e}")
print(f"  b2 (K) = {model3.coef_[1]:.10e}")
print(f"  b3 (N) = {model3.coef_[2]:.10e}")
print(f"  b4 (M*K) = {model3.coef_[3]:.10e}")
print(f"  b5 (M*N) = {model3.coef_[4]:.10e}")
print(f"  b6 (K*N) = {model3.coef_[5]:.10e}")
print(f"\nR² = {r2_score(y, y_pred3):.6f}")
print(f"MAE = {mean_absolute_error(y, y_pred3):.10e}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_pred3)):.10e}")

# ============================================================================
# Summary Comparison
# ============================================================================
print("\n" + "=" * 80)
print("Model Comparison Summary")
print("=" * 80)
print(f"{'Model':<40} {'R²':<12} {'MAE':<15} {'RMSE':<15}")
print("-" * 80)
print(f"{'Model 1: Basic (M,K,N)':<40} {r2_score(y, y_pred1):<12.6f} {mean_absolute_error(y, y_pred1):<15.6e} {np.sqrt(mean_squared_error(y, y_pred1)):<15.6e}")
print(f"{'Model 2: +M*K, +K*N':<40} {r2_score(y, y_pred2):<12.6f} {mean_absolute_error(y, y_pred2):<15.6e} {np.sqrt(mean_squared_error(y, y_pred2)):<15.6e}")
print(f"{'Model 3: +M*K, +M*N, +K*N':<40} {r2_score(y, y_pred3):<12.6f} {mean_absolute_error(y, y_pred3):<15.6e} {np.sqrt(mean_squared_error(y, y_pred3)):<15.6e}")

# ============================================================================
# Visualization
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models = [
    (y_pred1, "Model 1: Basic (M,K,N)", r2_score(y, y_pred1)),
    (y_pred2, "Model 2: +M×K, +K×N", r2_score(y, y_pred2)),
    (y_pred3, "Model 3: All interactions", r2_score(y, y_pred3))
]

for idx, (y_pred, title, r2) in enumerate(models):
    # Predicted vs Actual
    ax_pred = axes[0, idx]
    ax_pred.scatter(y, y_pred, alpha=0.5, s=10)
    ax_pred.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax_pred.set_xlabel('Actual latency_per_cycle')
    ax_pred.set_ylabel('Predicted latency_per_cycle')
    ax_pred.set_title(f'{title}\nR² = {r2:.6f}')
    ax_pred.grid(True, alpha=0.3)

    # Residuals
    ax_res = axes[1, idx]
    residuals = y - y_pred
    ax_res.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax_res.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax_res.set_xlabel('Predicted latency_per_cycle')
    ax_res.set_ylabel('Residuals')
    ax_res.set_title(f'Residual Plot')
    ax_res.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mkn_regression_comparison.png', dpi=150)
print(f"\nGraph saved as 'mkn_regression_comparison.png'")

# ============================================================================
# Best Model Selection
# ============================================================================
print("\n" + "=" * 80)
r2_scores = [r2_score(y, y_pred1), r2_score(y, y_pred2), r2_score(y, y_pred3)]
best_model_idx = np.argmax(r2_scores)
model_names = ["Model 1 (Basic)", "Model 2 (M*K, K*N)", "Model 3 (All interactions)"]
print(f"BEST MODEL: {model_names[best_model_idx]} with R² = {r2_scores[best_model_idx]:.6f}")
print("=" * 80)
