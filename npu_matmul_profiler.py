"""
NPU Matrix Multiplication Latency Profiling
Measures latency for various matrix dimensions considering NPU constraints:
- PE (Processing Element): 128 x 128
- Moving Tensor: 128 x 512

Uses AWS Neuron SDK for profiling on Inferentia2/Trainium instances.
"""

import time
import csv
import numpy as np
import torch
import torch_neuronx
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import os
import warnings


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
    def __init__(self, npu_config: NPUConfig = None, use_neuron: bool = True):
        """
        Initialize NPU Matmul Profiler

        Args:
            npu_config: NPU hardware configuration
            use_neuron: If True, use Neuron device. If False, use CPU (for testing)
        """
        self.npu_config = npu_config or NPUConfig()
        self.use_neuron = use_neuron

        # Set device
        if use_neuron:
            try:
                # Check if Neuron device is available
                self.device = torch.device('xla')
                print(f"Using Neuron device (XLA)")
            except Exception as e:
                warnings.warn(f"Neuron device not available: {e}. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.use_neuron = False
        else:
            self.device = torch.device('cpu')
            print("Using CPU device")

        # Cache for compiled models: key=(M,K,N), value=compiled_model
        self._compiled_models = {}

    def calculate_tiling(self, matmul_config: MatmulConfig) -> Tuple[int, int, int]:
        """
        Calculate number of tiles needed for each dimension
        Returns: (num_tiles_M, num_tiles_K, num_tiles_N)
        """
        num_tiles_M = (matmul_config.M + self.npu_config.PE_M - 1) // self.npu_config.PE_M
        num_tiles_N = (matmul_config.N + self.npu_config.PE_N - 1) // self.npu_config.PE_N
        num_tiles_K = (matmul_config.K + self.npu_config.MAX_TENSOR_K - 1) // self.npu_config.MAX_TENSOR_K

        return num_tiles_M, num_tiles_K, num_tiles_N

    def _get_or_compile_model(self, M: int, K: int, N: int):
        """
        Get compiled model from cache or compile new one

        Args:
            M, K, N: Matrix dimensions

        Returns:
            Compiled Neuron model or regular PyTorch function
        """
        key = (M, K, N)

        if key in self._compiled_models:
            return self._compiled_models[key]

        # Define a simple matmul module
        class MatmulModule(torch.nn.Module):
            def forward(self, a, b):
                return torch.matmul(a, b)

        model = MatmulModule()
        model.eval()

        if self.use_neuron:
            # Create example inputs for tracing
            example_a = torch.randn(M, K, dtype=torch.float32)
            example_b = torch.randn(K, N, dtype=torch.float32)

            try:
                # Compile for Neuron using torch_neuronx.trace
                print(f"  [Compiling for Neuron] M={M}, K={K}, N={N}... ", end='', flush=True)
                compiled_model = torch_neuronx.trace(
                    model,
                    (example_a, example_b),
                    compiler_workdir=f"/tmp/neuron_cache/matmul_{M}x{K}x{N}"
                )
                print("Done")
                self._compiled_models[key] = compiled_model
                return compiled_model
            except Exception as e:
                warnings.warn(f"Failed to compile model for shape ({M},{K},{N}): {e}")
                self._compiled_models[key] = model
                return model
        else:
            # CPU mode: just return the model
            self._compiled_models[key] = model
            return model

    def run_npu_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run matrix multiplication on NPU and measure latency

        Args:
            A: Input tensor (M, K)
            B: Input tensor (K, N)

        Returns:
            C: Output tensor (M, N)
            latency_ms: Latency in milliseconds
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Dimension mismatch: A.shape[1]={K} != B.shape[0]={K2}"

        # Get or compile model for this shape
        model = self._get_or_compile_model(M, K, N)

        # For Neuron: move tensors to XLA device if needed
        if self.use_neuron and str(A.device) != 'xla:0':
            A = A.to(self.device)
            B = B.to(self.device)

        # Measure latency
        start_time = time.perf_counter()
        C = model(A, B)

        # For XLA: need to mark step and synchronize
        if self.use_neuron:
            import torch_xla.core.xla_model as xm
            xm.mark_step()  # Trigger execution
            xm.wait_device_ops()  # Wait for completion

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
        # Create random tensors (on CPU initially)
        A = torch.randn(matmul_config.M, matmul_config.K, dtype=torch.float32)
        B = torch.randn(matmul_config.K, matmul_config.N, dtype=torch.float32)

        # Warmup
        for _ in range(warmup):
            _, _ = self.run_npu_matmul(A, B)

        # Measure
        latencies = []
        for _ in range(iterations):
            _, latency = self.run_npu_matmul(A, B)
            latencies.append(latency)

        avg_latency = float(np.mean(latencies))

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
    # Initialize profiler with Neuron device
    profiler = NPUMatmulProfiler(use_neuron=True)

    # Run different profiling suites
    output_dir = "./results"

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
