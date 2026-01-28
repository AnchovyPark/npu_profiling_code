import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv('maxsize_tile.csv')

# N이 512 이하인 데이터만 필터링
df_filtered = df[df['N'] <= 512].copy()

# K 값 (고정)
K = 128

# 독립 변수: (N - K)
X = (df_filtered['N'] - K).values.reshape(-1, 1)

# 종속 변수: latency_ms
y = df_filtered['latency_ms'].values

# 선형 회귀 분석
model = LinearRegression()
model.fit(X, y)

# 계수 추출
gamma = model.coef_[0]  # 기울기 = γ
alpha = model.intercept_  # 절편 = α

# 결과 출력
print("=" * 60)
print("선형 회귀 분석 결과")
print("=" * 60)
print(f"방정식: Latency = α + (N - K) * γ")
print(f"K = {K}")
print(f"\nα (상수항) = {alpha:.6f} ms")
print(f"γ (기울기) = {gamma:.6f} ms/cycle")
print(f"\nR² (결정계수) = {model.score(X, y):.6f}")
print("=" * 60)

# 예측값 계산
y_pred = model.predict(X)

# 잔차 분석
residuals = y - y_pred
print(f"\n평균 절대 오차 (MAE) = {np.mean(np.abs(residuals)):.6f} ms")
print(f"평균 제곱근 오차 (RMSE) = {np.sqrt(np.mean(residuals**2)):.6f} ms")

# 예측 예시
print("\n" + "=" * 60)
print("예측 예시:")
print("=" * 60)
for n_test in [136, 256, 384, 512]:
    predicted = alpha + (n_test - K) * gamma
    actual_row = df_filtered[df_filtered['N'] == n_test]
    if not actual_row.empty:
        actual = actual_row['latency_ms'].values[0]
        error = predicted - actual
        print(f"N={n_test}: 예측={predicted:.6f} ms, 실제={actual:.6f} ms, 오차={error:.6f} ms")

# 시각화
plt.figure(figsize=(12, 5))

# 1. 실제값 vs 예측값
plt.subplot(1, 2, 1)
plt.scatter(df_filtered['N'], y, label='실제값', alpha=0.6)
plt.plot(df_filtered['N'], y_pred, 'r-', label='예측값 (회귀선)', linewidth=2)
plt.xlabel('N')
plt.ylabel('Latency (ms)')
plt.title(f'Latency = {alpha:.4f} + (N - {K}) * {gamma:.6f}')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 잔차 플롯
plt.subplot(1, 2, 2)
plt.scatter(df_filtered['N'], residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('N')
plt.ylabel('잔차 (Residual)')
plt.title('잔차 분석')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latency_regression_analysis.png', dpi=150)
print("\n그래프가 'latency_regression_analysis.png'로 저장되었습니다.")
