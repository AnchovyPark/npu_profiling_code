"""
NPU Matrix Multiplication Latency Profiling
Measures latency for various matrix dimensions considering NPU constraints:
- PE (Processing Element): 128 x 128
- Moving Tensor: 128 x 512
"""

import time
import csv
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
import os


@dataclass
class NPUConfig:
    """NPU Hardware Configuration"""
    PE_M: int = 128  # PE height
    PE_N: int = 128  # PE width
    MAX_TENSOR_M: int = 128  # Max moving tensor height
    MAX_TENSOR_K: int = 512  # Max moving tensor width


@dataclass
class MatmulConfig:
    """Matrix Multiplication Configuration"""
    M: int  # Output height
    K: int  # Contraction dimension
    N: int  # Output width

    def __str__(self):
        return f"M={self.M}, K={self.K}, N={self.N}"


@dataclass
class ProfilingResult:
    """Profiling result for a single matmul"""
    M: int
    K: int
    N: int
    latency_ms: float
    num_tiles_M: int
    num_tiles_K: int
    num_tiles_N: int
    total_tiles: int

    def to_dict(self) -> Dict:
        return {
            'M': self.M,
            'K': self.K,
            'N': self.N,
            'latency_ms': self.latency_ms,
            'num_tiles_M': self.num_tiles_M,
            'num_tiles_K': self.num_tiles_K,
            'num_tiles_N': self.num_tiles_N,
            'total_tiles': self.total_tiles,
        }


class NPUMatmulProfiler:
    def __init__(self, npu_config: NPUConfig = None):
        self.npu_config = npu_config or NPUConfig()

    def calculate_tiling(self, matmul_config: MatmulConfig) -> Tuple[int, int, int]:
        """
        Calculate number of tiles needed for each dimension
        Returns: (num_tiles_M, num_tiles_K, num_tiles_N)
        """
        num_tiles_M = (matmul_config.M + self.npu_config.PE_M - 1) // self.npu_config.PE_M
        num_tiles_N = (matmul_config.N + self.npu_config.PE_N - 1) // self.npu_config.PE_N
        num_tiles_K = (matmul_config.K + self.npu_config.MAX_TENSOR_K - 1) // self.npu_config.MAX_TENSOR_K

        return num_tiles_M, num_tiles_K, num_tiles_N

    def run_npu_matmul(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run matrix multiplication on NPU and measure latency

        Args:
            A: Input matrix (M, K)
            B: Input matrix (K, N)

        Returns:
            C: Output matrix (M, N)
            latency_ms: Latency in milliseconds
        """
        # TODO: Replace with actual NPU API call
        # This is a placeholder using NumPy for demonstration

        start_time = time.perf_counter()

        # Placeholder: actual NPU matmul would be called here
        # Example NPU API calls might look like:
        # - torch_npu.matmul(A, B) for PyTorch NPU
        # - tflite inference for TensorFlow Lite
        # - Custom NPU runtime API

        C = np.matmul(A, B)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return C, latency_ms

    def profile_single_matmul(self, matmul_config: MatmulConfig, warmup: int = 3, iterations: int = 10) -> ProfilingResult:
        """
        Profile a single matrix multiplication configuration

        Args:
            matmul_config: Matrix dimensions
            warmup: Number of warmup iterations
            iterations: Number of measurement iterations

        Returns:
            ProfilingResult with latency and tiling information
        """
        # Create random matrices
        A = np.random.randn(matmul_config.M, matmul_config.K).astype(np.float32)
        B = np.random.randn(matmul_config.K, matmul_config.N).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            _, _ = self.run_npu_matmul(A, B)

        # Measure
        latencies = []
        for _ in range(iterations):
            _, latency = self.run_npu_matmul(A, B)
            latencies.append(latency)

        avg_latency = np.mean(latencies)

        # Calculate tiling
        num_tiles_M, num_tiles_K, num_tiles_N = self.calculate_tiling(matmul_config)
        total_tiles = num_tiles_M * num_tiles_K * num_tiles_N

        return ProfilingResult(
            M=matmul_config.M,
            K=matmul_config.K,
            N=matmul_config.N,
            latency_ms=avg_latency,
            num_tiles_M=num_tiles_M,
            num_tiles_K=num_tiles_K,
            num_tiles_N=num_tiles_N,
            total_tiles=total_tiles
        )

    def generate_matmul_configs(self, config_type: str = 'llm') -> List[MatmulConfig]:
        """
        Generate matrix multiplication configurations for profiling

        Args:
            config_type: Type of configurations
                - 'llm': LLM inference typical dimensions
                - 'sweep': Systematic sweep of dimensions
                - 'tiling': Focus on tiling boundaries
        """
        configs = []

        if config_type == 'llm':
            # LLM inference typical dimensions
            # batch_size * seq_len, hidden_dim, hidden_dim/intermediate_dim

            batch_sizes = [1, 2, 4, 8, 16, 32]
            seq_lens = [1, 64, 128, 256, 512, 1024, 2048]
            hidden_dims = [768, 1024, 2048, 4096, 8192]

            # Prefill phase: varying sequence lengths
            for bs in batch_sizes:
                for seq_len in seq_lens:
                    for hidden_dim in hidden_dims:
                        M = bs * seq_len
                        configs.append(MatmulConfig(M=M, K=hidden_dim, N=hidden_dim))
                        # FFN intermediate dimension (typically 4x hidden)
                        configs.append(MatmulConfig(M=M, K=hidden_dim, N=hidden_dim * 4))
                        configs.append(MatmulConfig(M=M, K=hidden_dim * 4, N=hidden_dim))

            # Decode phase: single token
            for bs in batch_sizes:
                for hidden_dim in hidden_dims:
                    M = bs
                    configs.append(MatmulConfig(M=M, K=hidden_dim, N=hidden_dim))
                    configs.append(MatmulConfig(M=M, K=hidden_dim, N=hidden_dim * 4))
                    configs.append(MatmulConfig(M=M, K=hidden_dim * 4, N=hidden_dim))

        elif config_type == 'sweep':
            # Systematic sweep
            M_dims = [64, 128, 256, 512, 1024, 2048, 4096]
            K_dims = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
            N_dims = [64, 128, 256, 512, 1024, 2048, 4096]

            for M in M_dims:
                for K in K_dims:
                    for N in N_dims:
                        configs.append(MatmulConfig(M=M, K=K, N=N))

        elif config_type == 'tiling':
            # Focus on tiling boundaries (PE: 128x128, Tensor: 128x512)
            PE_M = self.npu_config.PE_M
            PE_N = self.npu_config.PE_N
            MAX_K = self.npu_config.MAX_TENSOR_K

            # Test around tile boundaries
            boundary_offsets = [-32, -16, 0, 16, 32]
            multipliers = [1, 2, 3, 4]

            for mult_M in multipliers:
                for mult_N in multipliers:
                    for mult_K in multipliers:
                        for offset_M in boundary_offsets:
                            for offset_N in boundary_offsets:
                                for offset_K in boundary_offsets:
                                    M = mult_M * PE_M + offset_M
                                    N = mult_N * PE_N + offset_N
                                    K = mult_K * MAX_K + offset_K

                                    if M > 0 and N > 0 and K > 0:
                                        configs.append(MatmulConfig(M=M, K=K, N=N))

        return configs

    def save_results_to_csv(self, results: List[ProfilingResult], output_path: str):
        """Save profiling results to CSV file"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            if results:
                fieldnames = list(results[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result.to_dict())

        print(f"Results saved to {output_path}")

    def run_profiling_suite(self, config_type: str = 'llm', output_dir: str = './results',
                           warmup: int = 3, iterations: int = 10):
        """
        Run complete profiling suite

        Args:
            config_type: Type of configurations to test
            output_dir: Directory to save results
            warmup: Number of warmup iterations
            iterations: Number of measurement iterations
        """
        print(f"Generating {config_type} configurations...")
        configs = self.generate_matmul_configs(config_type)
        print(f"Total configurations: {len(configs)}")

        results = []
        for i, config in enumerate(configs):
            print(f"Profiling [{i+1}/{len(configs)}] {config}")
            result = self.profile_single_matmul(config, warmup=warmup, iterations=iterations)
            print(f"  Latency: {result.latency_ms:.4f} ms, Tiles: {result.total_tiles} "
                  f"(M:{result.num_tiles_M} x K:{result.num_tiles_K} x N:{result.num_tiles_N})")
            results.append(result)

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"npu_profiling_{config_type}_{timestamp}.csv")
        self.save_results_to_csv(results, output_path)

        return results


def main():
    """Main profiling script"""
    profiler = NPUMatmulProfiler()

    # Run different profiling suites
    output_dir = "/home/jhpark/pjh_project/kernel_npu_profiling/results"

    # 1. LLM-focused profiling
    print("="*80)
    print("Running LLM-focused profiling...")
    print("="*80)
    profiler.run_profiling_suite(config_type='llm', output_dir=output_dir, warmup=3, iterations=10)

    # 2. Tiling boundary profiling
    print("\n" + "="*80)
    print("Running tiling boundary profiling...")
    print("="*80)
    profiler.run_profiling_suite(config_type='tiling', output_dir=output_dir, warmup=3, iterations=10)

    # 3. Full sweep (warning: can be very large)
    # Uncomment if needed
    # print("\n" + "="*80)
    # print("Running full sweep profiling...")
    # print("="*80)
    # profiler.run_profiling_suite(config_type='sweep', output_dir=output_dir, warmup=3, iterations=10)


if __name__ == "__main__":
    main()
