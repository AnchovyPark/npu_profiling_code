"""
Example usage of NPU profiler
"""

from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig, NPUConfig


def example_1_single_matmul():
    """Example 1: Profile a single matrix multiplication"""
    print("\n" + "="*80)
    print("Example 1: Single Matrix Multiplication Profiling")
    print("="*80)

    profiler = NPUMatmulProfiler()

    # LLM prefill: batch_size=4, seq_len=512, hidden_dim=4096
    config = MatmulConfig(M=4*512, K=4096, N=4096)

    print(f"\nProfiling configuration: {config}")
    result = profiler.profile_single_matmul(config, warmup=5, iterations=20)

    print(f"\nResults:")
    print(f"  Latency: {result.latency_ms:.4f} ms")
    print(f"  Tiles: M={result.num_tiles_M}, K={result.num_tiles_K}, N={result.num_tiles_N}")
    print(f"  Total tiles: {result.total_tiles}")


def example_2_custom_configs():
    """Example 2: Profile custom configurations"""
    print("\n" + "="*80)
    print("Example 2: Custom Configuration List")
    print("="*80)

    profiler = NPUMatmulProfiler()

    # Define custom configurations
    configs = [
        # Decode phase: batch_size=1, hidden_dim=4096
        MatmulConfig(M=1, K=4096, N=4096),
        MatmulConfig(M=1, K=4096, N=16384),  # FFN

        # Decode phase: batch_size=8
        MatmulConfig(M=8, K=4096, N=4096),
        MatmulConfig(M=8, K=4096, N=16384),

        # Prefill phase: batch_size=1, seq_len=1024
        MatmulConfig(M=1024, K=4096, N=4096),
        MatmulConfig(M=1024, K=4096, N=16384),
    ]

    results = []
    for config in configs:
        print(f"\nProfiling {config}")
        result = profiler.profile_single_matmul(config, warmup=3, iterations=10)
        print(f"  Latency: {result.latency_ms:.4f} ms, Tiles: {result.total_tiles}")
        results.append(result)

    # Save to CSV
    output_path = "./results/custom_example.csv"
    profiler.save_results_to_csv(results, output_path)
    print(f"\nResults saved to {output_path}")


def example_3_tiling_boundary():
    """Example 3: Test tiling boundary effects"""
    print("\n" + "="*80)
    print("Example 3: Tiling Boundary Effects")
    print("="*80)

    profiler = NPUMatmulProfiler()

    # Test around PE boundary (128x128)
    M_values = [120, 128, 136, 248, 256, 264]
    K = 4096
    N = 4096

    results = []
    for M in M_values:
        config = MatmulConfig(M=M, K=K, N=N)
        result = profiler.profile_single_matmul(config, warmup=3, iterations=10)
        print(f"M={M:4d}: Latency={result.latency_ms:8.4f} ms, "
              f"Tiles_M={result.num_tiles_M}, Total_tiles={result.total_tiles}")
        results.append(result)

    # Analyze boundary effect
    print("\nBoundary analysis:")
    for i in range(len(results) - 1):
        if results[i+1].num_tiles_M > results[i].num_tiles_M:
            latency_increase = results[i+1].latency_ms - results[i].latency_ms
            pct_increase = (latency_increase / results[i].latency_ms) * 100
            print(f"  Crossing boundary at M={M_values[i]} -> {M_values[i+1]}: "
                  f"+{latency_increase:.4f} ms ({pct_increase:+.2f}%)")


def example_4_llm_inference_scenarios():
    """Example 4: Typical LLM inference scenarios"""
    print("\n" + "="*80)
    print("Example 4: LLM Inference Scenarios")
    print("="*80)

    profiler = NPUMatmulProfiler()

    scenarios = {
        "Prefill (bs=1, seq=128, h=4096)": MatmulConfig(M=1*128, K=4096, N=4096),
        "Prefill (bs=1, seq=2048, h=4096)": MatmulConfig(M=1*2048, K=4096, N=4096),
        "Prefill (bs=4, seq=512, h=4096)": MatmulConfig(M=4*512, K=4096, N=4096),

        "Decode (bs=1, h=4096)": MatmulConfig(M=1, K=4096, N=4096),
        "Decode (bs=8, h=4096)": MatmulConfig(M=8, K=4096, N=4096),
        "Decode (bs=32, h=4096)": MatmulConfig(M=32, K=4096, N=4096),

        "FFN Up (bs=1, seq=512, 4096->16384)": MatmulConfig(M=1*512, K=4096, N=16384),
        "FFN Down (bs=1, seq=512, 16384->4096)": MatmulConfig(M=1*512, K=16384, N=4096),
    }

    results = []
    for name, config in scenarios.items():
        print(f"\n{name}")
        print(f"  Config: {config}")
        result = profiler.profile_single_matmul(config, warmup=3, iterations=10)
        print(f"  Latency: {result.latency_ms:.4f} ms")
        print(f"  Tiles: {result.total_tiles} (M:{result.num_tiles_M} x K:{result.num_tiles_K} x N:{result.num_tiles_N})")
        results.append(result)

    output_path = "./results/llm_scenarios.csv"
    profiler.save_results_to_csv(results, output_path)


def example_5_quick_test():
    """Example 5: Quick test with small configurations"""
    print("\n" + "="*80)
    print("Example 5: Quick Test (Small Configurations)")
    print("="*80)

    profiler = NPUMatmulProfiler()

    # Small configurations for quick testing
    configs = [
        MatmulConfig(M=64, K=128, N=128),
        MatmulConfig(M=128, K=128, N=128),
        MatmulConfig(M=128, K=256, N=256),
        MatmulConfig(M=256, K=512, N=512),
    ]

    print("\nRunning quick tests...")
    for config in configs:
        result = profiler.profile_single_matmul(config, warmup=2, iterations=5)
        print(f"{config}: {result.latency_ms:.4f} ms (tiles: {result.total_tiles})")


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Single matmul", example_1_single_matmul),
        "2": ("Custom configs", example_2_custom_configs),
        "3": ("Tiling boundary", example_3_tiling_boundary),
        "4": ("LLM scenarios", example_4_llm_inference_scenarios),
        "5": ("Quick test", example_5_quick_test),
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            _, func = examples[example_num]
            func()
        else:
            print(f"Unknown example number: {example_num}")
            print("Available examples:")
            for num, (desc, _) in examples.items():
                print(f"  {num}: {desc}")
    else:
        print("Available examples:")
        for num, (desc, _) in examples.items():
            print(f"  {num}: {desc}")
        print("\nUsage: python example_usage.py <example_number>")
        print("\nRunning all examples...")

        for num, (desc, func) in examples.items():
            func()
