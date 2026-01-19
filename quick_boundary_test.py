"""
PE Boundary 효과 분석 스크립트
M dimension을 변화시키면서 boundary crossing 효과 측정
"""

from npu_matmul_profiler import NPUMatmulProfiler, MatmulConfig

def test_pe_boundary():
    """PE boundary (128) 효과 테스트"""
    profiler = NPUMatmulProfiler(use_neuron=True)

    print('='*80)
    print('PE Boundary 효과 분석 (M dimension)')
    print('='*80)
    print('PE size = 128, K=512 고정, N=512 고정')
    print('M을 변화시키면서 boundary crossing 효과 측정')
    print()

    # M dimension을 변화시키면서 테스트
    M_values = [120, 128, 136, 248, 256, 264]
    K = 512
    N = 512

    print(f'{"M":>4s} | {"Tiles_M":>7s} | {"Total":>5s} | {"Latency (ms)":>12s} | {"vs prev":>10s} | ')
    print('-'*70)

    results = []
    prev_latency = None

    for M in M_values:
        config = MatmulConfig(M=M, K=K, N=N)
        result = profiler.profile_single_matmul(config, warmup=2, iterations=5)
        results.append(result)

        if prev_latency is not None:
            change_pct = ((result.latency_ms - prev_latency) / prev_latency * 100)
            change_str = f'{change_pct:+6.1f}%'
        else:
            change_str = '-'

        marker = ''
        if M == 128:
            marker = ' <- boundary'
        elif M == 136:
            marker = ' <- +8 넘음!'
        elif M == 256:
            marker = ' <- boundary'
        elif M == 264:
            marker = ' <- +8 넘음!'

        print(f'{M:4d} | {result.num_tiles_M:7d} | {result.total_tiles:5d} | {result.latency_ms:12.4f} | {change_str:>10s} |{marker}')
        prev_latency = result.latency_ms

    print()
    print('[관찰]')
    print(f'  - M=120 (1 tile): {results[0].latency_ms:.4f} ms')
    print(f'  - M=128 (1 tile): {results[1].latency_ms:.4f} ms')
    print(f'  - M=136 (2 tiles): {results[2].latency_ms:.4f} ms')
    print(f'    -> 단 8만 넘어도 2배 tile 필요!')
    print(f'  - M=248 (2 tiles): {results[3].latency_ms:.4f} ms')
    print(f'  - M=256 (2 tiles): {results[4].latency_ms:.4f} ms')
    print(f'  - M=264 (3 tiles): {results[5].latency_ms:.4f} ms')

    return results

def test_k_dimension():
    """K dimension boundary 효과 (MAX_TENSOR_K=512)"""
    profiler = NPUMatmulProfiler(use_neuron=True)

    print('\n')
    print('='*80)
    print('K Dimension Boundary 효과 분석')
    print('='*80)
    print('Max Tensor K = 512')
    print('M=128 고정, N=128 고정, K를 변화')
    print()

    K_values = [480, 512, 520, 1000, 1024, 1040]
    M = 128
    N = 128

    print(f'{"K":>4s} | {"Tiles_K":>7s} | {"Total":>5s} | {"Latency (ms)":>12s} | {"vs prev":>10s} | ')
    print('-'*70)

    results = []
    prev_latency = None

    for K in K_values:
        config = MatmulConfig(M=M, K=K, N=N)
        result = profiler.profile_single_matmul(config, warmup=2, iterations=5)
        results.append(result)

        if prev_latency is not None:
            change_pct = ((result.latency_ms - prev_latency) / prev_latency * 100)
            change_str = f'{change_pct:+6.1f}%'
        else:
            change_str = '-'

        marker = ''
        if K == 512:
            marker = ' <- boundary'
        elif K == 520:
            marker = ' <- +8 넘음!'
        elif K == 1024:
            marker = ' <- 2x boundary'

        print(f'{K:4d} | {result.num_tiles_K:7d} | {result.total_tiles:5d} | {result.latency_ms:12.4f} | {change_str:>10s} |{marker}')
        prev_latency = result.latency_ms

    print()
    print('[관찰]')
    print(f'  - K=512 (1 tile):  {results[1].latency_ms:.4f} ms')
    print(f'  - K=520 (2 tiles): {results[2].latency_ms:.4f} ms')
    print(f'    -> K dimension boundary 넘으면 latency 증가!')

    return results

def test_llm_scenario():
    """실제 LLM inference 시나리오"""
    profiler = NPUMatmulProfiler(use_neuron=True)

    print('\n')
    print('='*80)
    print('LLM Inference 시나리오 테스트')
    print('='*80)
    print()

    scenarios = {
        'Decode (bs=1)': MatmulConfig(M=1, K=4096, N=4096),
        'Decode (bs=8)': MatmulConfig(M=8, K=4096, N=4096),
        'Decode (bs=16)': MatmulConfig(M=16, K=4096, N=4096),
        'Prefill (bs=1, seq=128)': MatmulConfig(M=128, K=4096, N=4096),
        'Prefill (bs=1, seq=256)': MatmulConfig(M=256, K=4096, N=4096),
        'Prefill (bs=1, seq=512)': MatmulConfig(M=512, K=4096, N=4096),
    }

    results = []
    for name, config in scenarios.items():
        result = profiler.profile_single_matmul(config, warmup=2, iterations=5)
        results.append((name, result))
        print(f'{name:30s}: {result.latency_ms:8.4f} ms (tiles: {result.total_tiles:4d}, M:{result.num_tiles_M} K:{result.num_tiles_K} N:{result.num_tiles_N})')

    return results

if __name__ == '__main__':
    # 1. PE boundary 테스트
    test_pe_boundary()

    # 2. K dimension 테스트
    test_k_dimension()

    # 3. LLM 시나리오 테스트
    test_llm_scenario()

    print('\n')
    print('='*80)
    print('테스트 완료!')
    print('='*80)
