import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import glob

# matmul 폴더 내 모든 CSV 파일 찾기
csv_files = glob.glob('*.csv')
print(f"발견된 CSV 파일: {csv_files}")
print("=" * 80)

# 모든 데이터를 담을 리스트
all_data = []

# 각 CSV 파일 읽기
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        # cycles와 latency_per_cycle 컬럼이 있는지 확인
        if 'cycles' in df.columns and 'latency_per_cycle' in df.columns:
            df['source_file'] = csv_file
            all_data.append(df[['cycles', 'latency_per_cycle', 'source_file']])
            print(f"[OK] {csv_file}: {len(df)} rows")
        else:
            print(f"[SKIP] {csv_file}: cycles or latency_per_cycle column missing")
    except Exception as e:
        print(f"[ERROR] {csv_file}: {e}")

print("=" * 80)

# 모든 데이터 합치기
if not all_data:
    print("선형 회귀 분석을 수행할 데이터가 없습니다.")
    exit(1)

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n총 데이터 포인트: {len(combined_df)}")
print(f"cycles 범위: {combined_df['cycles'].min()} ~ {combined_df['cycles'].max()}")
print(f"latency_per_cycle 범위: {combined_df['latency_per_cycle'].min():.8f} ~ {combined_df['latency_per_cycle'].max():.8f}")

# 선형 회귀 분석
# 방정식: latency_per_cycle = α + cycles × γ
X = combined_df['cycles'].values.reshape(-1, 1)
y = combined_df['latency_per_cycle'].values

model = LinearRegression()
model.fit(X, y)

# 계수 추출
gamma = model.coef_[0]
alpha = model.intercept_

# 결과 출력
print("\n" + "=" * 80)
print("선형 회귀 분석 결과")
print("=" * 80)
print(f"방정식: latency_per_cycle = α + cycles × γ")
print(f"\nα (절편) = {alpha:.10f}")
print(f"γ (기울기) = {gamma:.10e}")
print(f"\nR² (결정계수) = {model.score(X, y):.6f}")
print("=" * 80)

# 예측값 계산
y_pred = model.predict(X)

# 잔차 분석
residuals = y - y_pred
print(f"\n평균 절대 오차 (MAE) = {np.mean(np.abs(residuals)):.10f}")
print(f"평균 제곱근 오차 (RMSE) = {np.sqrt(np.mean(residuals**2)):.10f}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 전체 데이터 scatter plot with regression line
ax1 = axes[0, 0]
ax1.scatter(combined_df['cycles'], y, alpha=0.5, s=10)
ax1.plot(combined_df['cycles'], y_pred, 'r-', linewidth=2, label='Regression Line')
ax1.set_xlabel('Cycles')
ax1.set_ylabel('Latency per Cycle')
ax1.set_title(f'latency_per_cycle = {alpha:.6e} + cycles × {gamma:.6e}\nR² = {model.score(X, y):.6f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 파일별로 색상을 다르게 한 scatter plot
ax2 = axes[0, 1]
colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
for i, csv_file in enumerate(csv_files):
    file_data = combined_df[combined_df['source_file'] == csv_file]
    ax2.scatter(file_data['cycles'], file_data['latency_per_cycle'],
               alpha=0.6, s=20, label=csv_file, color=colors[i])
ax2.plot(combined_df['cycles'], y_pred, 'r-', linewidth=2, label='Regression Line')
ax2.set_xlabel('Cycles')
ax2.set_ylabel('Latency per Cycle')
ax2.set_title('Data by Source File')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 잔차 플롯
ax3 = axes[1, 0]
ax3.scatter(combined_df['cycles'], residuals, alpha=0.5, s=10)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Cycles')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot')
ax3.grid(True, alpha=0.3)

# 4. 잔차 히스토그램
ax4 = axes[1, 1]
ax4.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.set_title('Residual Distribution')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cycles_latency_regression_all_files.png', dpi=150)
print(f"\n그래프가 'cycles_latency_regression_all_files.png'로 저장되었습니다.")

# 파일별 통계
print("\n" + "=" * 80)
print("파일별 통계")
print("=" * 80)
for csv_file in csv_files:
    file_data = combined_df[combined_df['source_file'] == csv_file]
    if len(file_data) > 0:
        print(f"\n[{csv_file}]")
        print(f"  데이터 포인트 수: {len(file_data)}")
        print(f"  cycles 범위: {file_data['cycles'].min()} ~ {file_data['cycles'].max()}")
        print(f"  latency_per_cycle 평균: {file_data['latency_per_cycle'].mean():.8f}")
        print(f"  latency_per_cycle 표준편차: {file_data['latency_per_cycle'].std():.8f}")
