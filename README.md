# DeepSeek Engram Component Benchmark

This repository contains benchmarking scripts for the **Engram** module, specifically focused on evaluating CPU-offloading strategies for large-scale language model inference. The script supports **NVIDIA GPU**, **Intel Gaudi (HPU)**, and **Pure CPU** environments.

The script measures the performance of N-gram hashing, Embedding lookups, Key/Value (KV) cache computation, and PCIe data transfer latencies.

## Benchmark Strategies

The script compares two primary workflow strategies for handling Engram layers (when an accelerator is available):

### Strategy 1: Compute KV on CPU
*   **Flow**: `Hash & Embed (CPU)` $\to$ `Compute KV (CPU)` $\to$ `Upload KV to Device`
*   **UseCase**: Maximizes Device memory savings by keeping the Embedding table and intermediate states on the Host RAM.
*   **Critical Path**: CPU Compute capability + Bandwidth required to upload the final Keys and Values.

### Strategy 2: Compute KV on Device (GPU/HPU)
*   **Flow**: `Hash & Embed (CPU)` $\to$ `Upload Embeddings to Device` $\to$ `Compute KV (Device)`
*   **UseCase**: Balances memory usage and speed; leverages accelerator for matrix multiplications (KV projection) while keeping the massive Embedding table on CPU.
*   **Critical Path**: Bandwidth required to upload Embeddings.

> **Note**: In a **Pure CPU** environment, transfer and end-to-end strategies are automatically disabled. Only `hash` and `compute` components are benchmarked, and pinned memory is disabled.

## Usage

### 1. Basic Run
Run the full benchmark. The script automatically detects CUDA or HPU devices.
```bash
python engram_benchmark.py
```

### 2. Custom Input Sizes
Test different batch sizes or sequence lengths:
```bash
python engram_benchmark.py --batch_size 1 --seq_len 8192
```

### 3. Run Specific Components
You can isolate specific parts of the pipeline using the `--cases` flag:
*   `hash`: Measure N-gram hashing and embedding lookup on CPU.
*   `compute`: Measure KV projection (Linear layers) on CPU.
*   `up_embed`: Measure PCIe transfer time for Embeddings (Requires GPU/HPU).
*   `up_kv`: Measure PCIe transfer time for Key/Value states (Requires GPU/HPU).
*   `full`: Run End-to-End latency tests (Requires GPU/HPU).

Example:
```bash
python engram_benchmark.py --cases up_embed up_kv --iterations 100
```

### 4. Advanced Features
Enable SGL kernels (if installed) or PyTorch Profiler:
```bash
python engram_benchmark.py --enable_sgl_kernel --enable_profiler
```

## Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--cases` | `all` | Specific benchmark cases to run (`hash`, `compute`, `up_embed`, `up_kv`, `full`). |
| `--iterations` | `50` | Number of iterations for averaging results. |
| `--batch_size` | `64` | Input batch size. |
| `--seq_len` | `4096` | Input sequence length. |
| `--dtype` | `bfloat16` | Data type (`float32`, `float16`, `bfloat16`). |
| `--enable_profiler` | `False` | Enable PyTorch Profiler for detailed trace analysis. |
| `--enable_sgl_kernel` | `False` | Enable optimized SGL kernels. |

## Requirements
*   Python 3.8+
*   PyTorch (CUDA or HPU support recommended)
*   Transformers
*   SymPy
*   NumPy
*   (Optional) sgl-kernel
