"""
================================================================================
[Engram Architecture CPU-Offload Benchmark]

DESCRIPTION:
1. Purpose:
   This script benchmarks the latency and bandwidth of the Engram module components, 
   specifically evaluating trade-offs between CPU and GPU execution strategies for 
   memory-constrained inference.

2. Strategies Evaluated:
   a) Strategy 1 (KV on CPU): 
      - Workflow: N-gram Hashing (CPU) -> Embedding Lookup (CPU) -> KV Projection (CPU) -> Upload KV to GPU.
      - Benefit: Saves GPU memory by offloading the large Embedding table and KV projections.
      - Bottleneck: PCIe bandwidth for uploading Keys/Values.
      
   b) Strategy 2 (KV on GPU):
      - Workflow: N-gram Hashing (CPU) -> Embedding Lookup (CPU) -> Upload Embeddings to GPU -> KV Projection (GPU).
      - Benefit: Faster computation of KV on GPU.
      - Bottleneck: PCIe bandwidth for uploading Embeddings (usually smaller than KV).

3. Metrics:
   - Latency (ms): Time taken for each step (hashing, lookup, compute, transfer).
   - Bandwidth (GB/s): Effective PCIe throughput during tensor transfers.
   - End-to-End Latency: Total time for the Engram forward pass in both strategies.

4. Usage:
   Run with default settings:
     python engram_benchmark.py
   
   Specific cases:
     python engram_benchmark.py --cases hash compute up_kv
     
   See help for details:
     python engram_benchmark.py --help
================================================================================
"""

## built-in
from typing import List
from dataclasses import dataclass, field
import math
import time
import argparse

## third-class
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

try:
    device = torch.cuda.current_device()
except:
    device = torch.device("hpu")
cpu_device = torch.device("cpu")

print(torch.backends.mkldnn.is_available())

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 1
    vocab_size: int = 129280
    num_layers: int = 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)
            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y
    
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        
        return output
    
class Engram(nn.Module):
    def __init__(self,layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size,backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size,backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
    
    def step1_hash_and_embed(self, input_ids):
        """
        Step 1: Hash input IDs and retrieve embeddings.
        Returns: embeddings [B, L, engram_hidden_size]
        """
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        return embeddings

    def step2_compute_kv(self, embeddings):
        """
        Step 2: Compute Key, Value, Gates.
        Returns: value [B, L, HC, D] (The signal before ShortConv)
        """
        keys = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            # normed_key = self.norm1[hc_idx](key)
            # keys.append(normed_key)
            keys.append(key)
        value = self.value_proj(embeddings).unsqueeze(2)
        return keys,value

    def forward(self, input_ids, compute_kv_loc="cpu", **kwargs):
        """
        Full forward pass combining steps.
        
        Args:
            input_ids: Input token IDs.
            compute_kv_loc (str): "cpu" means Key/Value computation happens on CPU, then uploaded.
                                  "gpu" means Embeddings are uploaded, then Key/Value computation happens on GPU.
        """
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        
        if compute_kv_loc == "gpu":
            # Scenario 2: Hash & Embed (CPU) -> Upload Embedding -> (KV Compute on GPU)
            return embeddings.to(device)

        # Scenario 1 (Default): Hash & Embed (CPU) -> Compute KV (CPU) -> Upload KV
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            _ = normed_key.to(device)
        value = self.value_proj(embeddings).unsqueeze(2)
        value_ongpu = value.to(device)
        return value_ongpu 

class TransformerBlock(nn.Module):
    def __init__(self,layer_id):
        super().__init__()
        self.attn = lambda x:x
        self.moe  = lambda x:x
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)
    
    def forward(self,input_ids,hidden_states):
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states,input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states

def benchmark_engram(target_cases=None,iterations=50,batch_size=64,seq_len=4096,dtype=torch.bfloat16):
    print(f"Preparing Engram Component Benchmark (CPU)...")
    
    # Settings
    target_layer_id = 1

    # Define available cases mapping with descriptive names
    all_cases = [
        "hash_embed_cpu",      # Step 1
        "upload_emb_to_gpu",   # Step 3
        "compute_kv_cpu",      # Step 2
        "upload_kv_to_gpu",    # Step 4
        "e2e_kv_on_cpu",       # Full Flow 1
        "e2e_kv_on_gpu"        # Full Flow 2
    ]
    
    if target_cases is None or "all" in target_cases:
        run_cases = all_cases
    else:
        run_cases = set(target_cases)

    print(f"Settings: Batch={batch_size}, SeqLen={seq_len}, hc_mult={backbone_config.hc_mult}, LayerID={target_layer_id}, Dtype={dtype}")
    print(f"Selected Cases: {run_cases}")
    print(f"Running detailed component benchmark ({iterations} iters)...")

    # Generate dummy data
    print("Generating dummy inputs...")
    input_ids = torch.randint(0, int(backbone_config.vocab_size*0.7), (batch_size, seq_len), device=cpu_device)

    # Initialize
    print("Initializing Engram Layer...")
    try:
        t0_init = time.time()
        engram_layer = Engram(layer_id=target_layer_id).to(dtype).to(cpu_device)
        print(f"Initialization finished in {time.time() - t0_init:.2f} seconds.")
    except Exception as e:
        print(f"Error initializing Engram layer: {e}")
        return

    # Pre-calculate references to avoid overhead in isolated tests
    print("Pre-calculating reference tensors for isolated tests...")
    with torch.no_grad():
        ref_embeddings = engram_layer.step1_hash_and_embed(input_ids)
        ref_embeddings = ref_embeddings.pin_memory()
        ref_keys, ref_value = engram_layer.step2_compute_kv(ref_embeddings)
        ref_keys = [k.pin_memory() for k in ref_keys]
        ref_value = ref_value.pin_memory()

    # Warmup
    print("Warming up (5 iters)...")
    with torch.no_grad():
        for _ in range(5):
            if "e2e_kv_on_cpu" in run_cases:
                _ = engram_layer(input_ids=input_ids, compute_kv_loc="cpu")
            if "e2e_kv_on_gpu" in run_cases:
                _ = engram_layer(input_ids=input_ids, compute_kv_loc="gpu")
            if "upload_emb_to_gpu" in run_cases:
               _ = ref_embeddings.to(device)

    
    results = {k: [] for k in all_cases}
    
    # Store sizes for reporting
    sizes_bytes = {
        "embedding": 0,
        "keys": 0,
        "value": 0
    }

    # Calculate sizes once (in Bytes)
    sizes_bytes["embedding"] = ref_embeddings.numel() * ref_embeddings.element_size()
    sizes_bytes["keys"] = sum([k.numel() for k in ref_keys]) * ref_keys[0].element_size()
    sizes_bytes["value"] = ref_value.numel() * ref_value.element_size()

    with torch.no_grad():
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...", end="\r")
            
            # Full Forward Baselines
            if "e2e_kv_on_cpu" in run_cases:
                time.sleep(0.005)
                t_start = time.perf_counter()
                _ = engram_layer(input_ids=input_ids, compute_kv_loc="cpu")
                results["e2e_kv_on_cpu"].append((time.perf_counter() - t_start) * 1000)

            if "e2e_kv_on_gpu" in run_cases:
                time.sleep(0.005)
                t_start = time.perf_counter()
                _ = engram_layer(input_ids=input_ids, compute_kv_loc="gpu")
                results["e2e_kv_on_gpu"].append((time.perf_counter() - t_start) * 1000)

            # Component Breakdown
            embeddings = ref_embeddings # Default to reference if Step 1 skipped
            
            # 1. Hash & Embedding
            # If we need to measure this step, run it. 
            # Otherwise use pre-calculated reference for subsequent steps to save time.
            if "hash_embed_cpu" in run_cases:
                time.sleep(0.005)
                t_start_1 = time.perf_counter()
                embeddings = engram_layer.step1_hash_and_embed(input_ids)
                t_end_1 = time.perf_counter()
                results["hash_embed_cpu"].append((t_end_1 - t_start_1) * 1000)
            
            if "upload_emb_to_gpu" in run_cases or "compute_kv_cpu" in run_cases or "upload_kv_to_gpu" in run_cases:
                # 3. Upload Embedding
                if "upload_emb_to_gpu" in run_cases:
                    time.sleep(0.005)
                    t_start_up1 = time.perf_counter()
                    _ = ref_embeddings.to(device, non_blocking=False)
                    t_end_up1 = time.perf_counter()
                    results["upload_emb_to_gpu"].append((t_end_up1 - t_start_up1) * 1000)
                
                # 2. Key & Value Calculation
                if "compute_kv_cpu" in run_cases:
                    time.sleep(0.005)
                    t_start_2 = time.perf_counter()
                    keys, value = engram_layer.step2_compute_kv(embeddings)
                    t_end_2 = time.perf_counter()
                    results["compute_kv_cpu"].append((t_end_2 - t_start_2) * 1000)

                # 4. Upload Key & Value (Simulate)
                if "upload_kv_to_gpu" in run_cases:
                    time.sleep(0.005)
                    t_start_up2 = time.perf_counter()
                    for key in ref_keys:
                        _ = key.to(device,non_blocking=False)
                    _ = ref_value.to(device,non_blocking=False)
                    t_end_up2 = time.perf_counter()
                    results["upload_kv_to_gpu"].append((t_end_up2 - t_start_up2) * 1000)

    # Reporting
    print("\n" + "="*90)
    print(" ENGRAM COMPONENT BENCHMARK RESULTS")
    print("="*90)
    print(f" Input Shape: ({batch_size}, {seq_len}) | Iterations: {iterations}")
    
    
    def get_avg(key):
        if key not in results or not results[key]: return 0.0
        return sum(results[key]) / len(results[key])

    def print_metric(name, key, size_bytes=None):
        if key not in run_cases:
            return
            
        data = results[key]
        if not data:
            return
            
        avg_v = sum(data) / len(data)
        min_v = min(data)
        max_v = max(data)
        
        size_str = "-"
        bw_str = "-"

        if size_bytes is not None:
            # Display Size in MiB (1024^2)
            size_mib_val = size_bytes / (1024**2)
            size_str = f"{size_mib_val:<10.2f}"

            if avg_v > 0:
                # BW (GB/s) = Bytes / Seconds / 1024^9
                # Seconds = avg_v / 1000
                bw_val = size_bytes / (avg_v / 1000.0) / 1024**3 
                bw_str = f"{bw_val:<10.2f}"
            
        print(f"{name:<50} | {avg_v:<10.4f} | {min_v:<10.4f} | {max_v:<10.4f} | {size_str} | {bw_str}")

    # Group 1: Compute KV on CPU
    print("\n[Group 1: Strategy - Compute KV on CPU]")
    print("Description: Hash/Embed/KV-Compute are done on CPU. Only final KV tensors are uploaded to GPU.")
    print(f"{'(Flow: Hash&Embed -> Compute KV -> Upload KV)':<50} | Steps: 1, 2, 4")
    t1 = get_avg("hash_embed_cpu")
    t2 = get_avg("compute_kv_cpu")
    t4 = get_avg("upload_kv_to_gpu")
    kv_total_size = sizes_bytes["keys"] + sizes_bytes["value"]

    print("-" * 90)
    print(f"{'Metric':<50} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Size (MiB)':<10} | {'BW (GB/s)':<10}")
    print("-" * 90)
    print_metric("1. Hash & Embedding (CPU)", "hash_embed_cpu")
    print_metric("2. Compute KV (CPU)", "compute_kv_cpu")
    print_metric("4. Upload KV (CPU->GPU)", "upload_kv_to_gpu", kv_total_size)
    print("-" * 90)
    print(f"{'SUM of Components':<50} | {(t1+t2+t4):<10.4f}")
    print_metric("End-to-End forward (overhead for pin memory)", "e2e_kv_on_cpu")

    # Group 2: Compute KV on GPU
    print("\n[Group 2: Strategy - Compute KV on GPU]")
    print("Description: Hash/Embed done on CPU. Embeddings are uploaded. KV Compute happens on GPU.")
    print(f"{'(Flow: Hash&Embed -> Upload Embed -> [GPU Kernels])':<50} | Steps: 1, 3")
    t1 = get_avg("hash_embed_cpu")
    t3 = get_avg("upload_emb_to_gpu")

    print("-" * 90)
    print(f"{'Metric':<50} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Size (MiB)':<10} | {'BW (GB/s)':<10}")
    print("-" * 90)
    print_metric("1. Hash & Embedding (CPU)", "hash_embed_cpu")
    print_metric("3. Upload Embedding (CPU->GPU)", "upload_emb_to_gpu", sizes_bytes["embedding"])
    print("-" * 90)
    print(f"{'SUM of Components':<50} | {(t1+t3):<10.4f}")
    print_metric("End-to-End forward (overhead for pin memory)", "e2e_kv_on_gpu")

    print("="*90)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Engram Component")
    parser.add_argument(
        "--cases", 
        nargs="+", 
        choices=["hash", "compute", "up_embed", "up_kv", "full", "all"],
        default=["all"],
        help="Select specific benchmark cases to run."
    )
    parser.add_argument(
        "-i","--iterations",
        type=int,
        default=50,
        help="Number of iterations for each benchmark case."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the benchmark."
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Sequence length for the benchmark."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for the model."
    )
    args = parser.parse_args()

    # Map CLI args to internal keys
    cli_map = {
        "hash": "hash_embed_cpu",
        "compute": "compute_kv_cpu",
        "up_embed": "upload_emb_to_gpu",
        "up_kv": "upload_kv_to_gpu",
        "full": ["e2e_kv_on_cpu", "e2e_kv_on_gpu"],
        "all": "all"
    }
    
    target_cases = []
    if "all" in args.cases:
        target_cases = None
    else:
        for c in args.cases:
            val = cli_map[c]
            if isinstance(val, list):
                target_cases.extend(val)
            else:
                target_cases.append(val)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }

    benchmark_engram(
        target_cases=target_cases,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=dtype_map[args.dtype]
    )
