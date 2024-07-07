import numpy as cp
import numpy as np
import time

def check_gpu_config():
    print(f"CUDA available: {cp.cuda.is_available()}")
    print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"GPU device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Memory available: {cp.cuda.runtime.memGetInfo()[0] / 1e9:.2f} GB")

def benchmark_cpu(size, iterations):
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    times = []
    for _ in range(iterations):
        start_time = time.time()
        C = np.dot(A, B)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

def benchmark_gpu(size, iterations):
    A = cp.random.rand(size, size, dtype=cp.float32)
    B = cp.random.rand(size, size, dtype=cp.float32)
    times = []
    cp.cuda.Stream.null.synchronize()
    for _ in range(iterations):
        start_time = time.time()
        C = cp.dot(A, B)
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

# Fonction CUDA personnalisée pour une opération intensive
cuda_kernel = cp.RawKernel(r'''
extern "C" __global__
void intensive_op(const float* x, float* y, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float temp = x[tid];
        for (int i = 0; i < 1000; ++i) {
            temp = sinf(temp) * cosf(temp);
        }
        y[tid] = temp;
    }
}
''', 'intensive_op')

def benchmark_gpu_intensive(size, iterations):
    x = cp.random.rand(size * size, dtype=cp.float32)
    y = cp.zeros_like(x)
    times = []
    cp.cuda.Stream.null.synchronize()
    for _ in range(iterations):
        start_time = time.time()
        # Correction : Passer un tuple pour l'argument 'grid'
        grid = ((size * size + 255) // 256,)
        block = (256,)
        cuda_kernel(grid, block, (x, y, size * size))
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

def run_benchmarks(sizes, iterations):
    check_gpu_config()
    print("\nRunning benchmarks...")
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        cpu_time = benchmark_cpu(size, iterations)
        gpu_time = benchmark_gpu(size, iterations)
        gpu_intensive_time = benchmark_gpu_intensive(size, iterations)
        print(f"CPU time: {cpu_time:.6f} seconds")
        print(f"GPU time (matrix mult): {gpu_time:.6f} seconds")
        print(f"GPU time (intensive op): {gpu_intensive_time:.6f} seconds")
        print(f"GPU Speedup (matrix mult): {cpu_time / gpu_time:.2f}x")
        print(f"GPU Speedup (intensive op): {cpu_time / gpu_intensive_time:.2f}x")

if __name__ == "__main__":
    # Warm-up GPU
    cp.cuda.Stream.null.synchronize()
    x = cp.random.rand(1000, 1000, dtype=cp.float32)
    cp.dot(x, x)
    cp.cuda.Stream.null.synchronize()

    sizes = [1000, 5000]
    iterations = 5
    run_benchmarks(sizes, iterations)