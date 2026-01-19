"""
Analyze NPU tiling behavior from profiling results
Shows latency jumps at tile boundaries
"""

import pandas as pd
import sys

def analyze_tiling(csv_file):
    """Analyze tiling behavior from CSV results"""
    df = pd.read_csv(csv_file)

    print("=" * 80)
    print("NPU Tiling Analysis")
    print("=" * 80)

    # PE 크기
    PE_M, PE_N = 128, 128
    MAX_K = 512

    print(f"\nNPU Constraints:")
    print(f"  PE Size: {PE_M} x {PE_N}")
    print(f"  Max K: {MAX_K}")

    # Tile boundaries 계산
    df['tiles_M'] = ((df['M'] + PE_M - 1) // PE_M)
    df['tiles_K'] = ((df['K'] + MAX_K - 1) // MAX_K)
    df['tiles_N'] = ((df['N'] + PE_N - 1) // PE_N)
    df['total_tiles'] = df['tiles_M'] * df['tiles_K'] * df['tiles_N']

    # Latency per tile
    df['latency_per_tile'] = df['latency_ms'] / df['total_tiles']

    print(f"\n{'M':>6} {'K':>6} {'N':>6} | {'TilesM':>6} {'TilesK':>6} {'TilesN':>6} {'Total':>6} | {'Latency':>10} {'Per Tile':>10} | {'Jump':>8}")
    print("-" * 80)

    prev_latency = None
    for idx, row in df.iterrows():
        M, K, N = int(row['M']), int(row['K']), int(row['N'])
        tiles_M = int(row['tiles_M'])
        tiles_K = int(row['tiles_K'])
        tiles_N = int(row['tiles_N'])
        total_tiles = int(row['total_tiles'])
        latency = row['latency_ms']
        per_tile = row['latency_per_tile']

        # Boundary detection: tile 개수가 증가했는지 확인
        jump_marker = ""
        if prev_latency is not None:
            jump_ratio = latency / prev_latency
            if jump_ratio > 1.5:  # 50% 이상 증가
                jump_marker = f"↑{jump_ratio:.1f}x"

        # Boundary 표시
        boundary = ""
        if M % PE_M == 0 or K % MAX_K == 0 or N % PE_N == 0:
            boundary = " ←"

        print(f"{M:6d} {K:6d} {N:6d} | {tiles_M:6d} {tiles_K:6d} {tiles_N:6d} {total_tiles:6d} | "
              f"{latency:10.4f} {per_tile:10.4f} | {jump_marker:>8}{boundary}")

        prev_latency = latency

    print("\n" + "=" * 80)
    print("Legend:")
    print("  ← = Exact tile boundary (M%128=0, K%512=0, or N%128=0)")
    print("  ↑ = Significant latency jump (>1.5x from previous)")
    print("=" * 80)

    # Efficiency analysis
    print("\nEfficiency Analysis:")
    print(f"  Best latency per tile: {df['latency_per_tile'].min():.4f} ms")
    print(f"  Worst latency per tile: {df['latency_per_tile'].max():.4f} ms")
    print(f"  Ratio (worst/best): {df['latency_per_tile'].max() / df['latency_per_tile'].min():.2f}x")
    print()
    print("  → Penalty from inefficient tiling (non-boundary sizes)")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "./results/custom_test.csv"
    analyze_tiling(csv_file)
