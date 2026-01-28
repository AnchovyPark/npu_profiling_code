"""
Matmul Boundary Test
PE와 Moving Tensor boundary에서 latency 측정
"""

from profiler import run_profiling

# PE_K=128, MOVING_N=512의 배수로 측정
K_values = [128, 256, 384, 512]
M_values = [512, 640, 768, 896, 1024]
N_values = [1024, 1536, 2048, 2560, 3072, 3584, 4096]

run_profiling(M_values, K_values, N_values, '../results/matmul/K_variation_test.csv')
