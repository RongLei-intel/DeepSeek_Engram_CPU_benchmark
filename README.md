# DeepSeek Engram Component Benchmark

This repository contains benchmarking scripts for the **Engram** module, specifically focused on evaluating CPU-offloading strategies for large-scale language model inference.

The script measures the performance of N-gram hashing, Embedding lookups, Key/Value (KV) cache computation, and PCIe data transfer latencies.

## Benchmark Strategies

The script compares two primary workflow strategies for handling Engram layers:

### Strategy 1: Compute KV on CPU
*   **Flow**: `Hash & Embed (CPU)` $\to$ `Compute KV (CPU)` $\to$ `Upload KV to GPU`
*   **UseCase**: Maximizes GPU memory savings by keeping the Embedding table and intermediate states on the Host RAM.
*   **Critical Path**: CPU Compute capability + Bandwidth required to upload the final Keys and Values.

### Strategy 2: Compute KV on GPU
*   **Flow**: `Hash & Embed (CPU)` $\to$ `Upload Embeddings to GPU` $\to$ `Compute KV (GPU)`
*   **UseCase**: Balances memory usage and speed; leverages GPU for matrix multiplications (KV projection) while keeping the massive Embedding table on CPU.
*   **Critical Path**: Bandwidth required to upload Embeddings.

## Usage

### 1. Basic Run
Run the full benchmark with default settings (Batch Size 64, Sequence Length 4096):
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
*   `up_embed`: Measure PCIe transfer time for Embeddings.
*   `up_kv`: Measure PCIe transfer time for Key/Value states.
*   `full`: Run End-to-End latency tests.

Example:
```bash
python engram_benchmark.py --cases up_embed up_kv --iterations 100
```

## Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--cases` | `all` | Specific benchmark cases to run (`hash`, `compute`, `up_embed`, `up_kv`, `full`). |
| `--iterations` | `50` | Number of iterations for averaging results. |
| `--batch_size` | `64` | Input batch size. |
| `--seq_len` | `4096` | Input sequence length. |
| `--dtype` | `bfloat16` | Data type (`float32`, `float16`, `bfloat16`). |

## Requirements
*   Python 3.8+
*   PyTorch (with CUDA support)
*   Transformers
*   SymPy
*   NumPy
