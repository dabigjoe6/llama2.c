# Hand-written NEON matmul for llama2.c on an M1

This is my fork of [karpathy/llama2.c](https://github.com/karpathy/llama2.c). The original README is now [README_original.md](README_original.md).

I used it to learn how to profile a hot path and write performant code for it: SIMD, register tiling, memory layout, reading assembly. Everything here is about one function, `matmul`, running on **one core** of an 8 GB M1 MacBook Air.

Everything is compared against the compiler at -O3 and -Ofast. -Ofast already vectorises the original loop, so the main reason for doing it by hand is to learn what the compiler does and what gap it might leave on the table. I also wanted the speed to not depend on whether the compiler decides to vectorise, which is why hot paths are often hand written. `matmul_original` only gets fast with `-ffast-math`, and that changes float behaviour for the whole program.

> [!IMPORTANT]
> This is not production code. `matmul_simd` and `matmul_simd_packed` were written to learn, not with portability, error handling or best practices in mind, and they only run on ARM NEON.

## Results

M1 MacBook Air (8 GB), macOS 27, Apple clang 21, stories15M, single thread.

| kernel | W in cache, -O3 | W in cache, -Ofast | W from DRAM, -O3 | W from DRAM, -Ofast | tok/s, -O3 | tok/s, -Ofast |
|---|---|---|---|---|---|---|
| `matmul_original` (original loop) | 46,781 ns | 6,603 ns | 7.0 GB/s | 37.8 GB/s | 101 | 487 |
| `matmul_simd` (16-row NEON) | 4,939 ns | 4,997 ns | 18.1 GB/s | 17.2 GB/s | 218 | 232 |
| `matmul_simd_packed` (16-row NEON, packed W) | 4,218 ns | 4,342 ns | 44.3 GB/s | 40.0 GB/s | 475 | 496 |

- W in cache: 288 × 288, best of 15 trials, each trial averaging 5,000 calls.
- W from DRAM: 32000 × 288 (the real classifier), best of 15 trials, each trial averaging 50 calls. GB/s is bytes of W divided by time.
- tok/s: median of 7 runs, 256 tokens each.
- Against the -O3 build, `matmul_simd_packed` is about 4.7x the tok/s. Against -Ofast it's about 1.5x faster when W is in cache, and slightly faster when W comes from DRAM (40.0 vs 37.8 GB/s).
- Each kernel's output is checked against `matmul_original`, and all three generate the same story for the same seed.

## Running it

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
make runfast      # -Ofast build, or `make run` for -O3

./run stories15M.bin -s 49 -n 256          # matmul_simd (default)
./run stories15M.bin -s 49 -n 256 -u y     # matmul_original, the original scalar loop
./run stories15M.bin -s 49 -n 256 -k y     # matmul_simd_packed
./run stories15M.bin -h y                  # in-cache vs DRAM benchmark for all three
```

`-k y` packs a second copy of the weights at load time, about 61 MB more memory.

## Analysis

Profiled with Instruments: Time Profiler for where the time goes, and CPU Counters for IPC (instructions per cycle) and cache misses.

Time Profiler shows almost all the time is spent in `matmul`. Every matmul in the forward pass is a matrix × vector:

```
xout (d) = W (d × n) · x (n)
```

stories15M is dim 288, hidden dim 768, 6 layers, vocab 32000. One token reads:

| matrix | shape | size (fp32) |
|---|---|---|
| wq, wk, wv, wo | 288 × 288 each | 0.33 MB each |
| w1, w3 | 768 × 288 each | 0.88 MB each |
| w2 | 288 × 768 | 0.88 MB |
| per layer | | about 4.0 MB |
| 6 layers | | about 24 MB |
| wcls (classifier) | 32000 × 288 | about 37 MB |
| **per token** | | **about 61 MB** |

- **Each weight is read once per token and used once.** 61 MB is far more than the 12 MB L2, so every weight comes from DRAM on every token. Cache tiling can't help, there's no reuse for it to keep in cache.
- **x is reused for every row, and it's small.** At most 768 floats (3 KB), so it stays in the 128 KB L1.

So the misses on W can't be removed, only streamed as fast as possible, and x is already cheap. IPC for the original loop at -O2 is around 1.4 out of a possible ~8 on an M1 performance core. The -O3 build only moves W at 7 GB/s, far below what memory can deliver (about 68 GB/s on the M1), so it isn't waiting on memory. It's waiting on its own additions (see step 3). The problem is on the compute side.

## Optimisation

### 1. SIMD and splitting the registers

Since IPC is low and the loop is compute bound rather than memory bound, the next step is to do more work per instruction with SIMD (NEON, 4 floats per instruction), and to remove the dependency between iterations by unrolling into independent sums, so the core can run several FMAs at once (instruction level parallelism).

On the M1 an FMA (`fmla`) has a latency of 4 cycles, and 4 can start every cycle ([Dougall Johnson's Firestorm tables](https://dougallj.github.io/applecpu/firestorm-simd.html)). Keeping the FMA pipes busy takes 4 × 4 = **16 FMAs in flight**, so 16 independent sums. With a single sum, every FMA waits 4 cycles for the previous one.

ARM NEON has 32 registers (`v0` to `v31`), and every sum, x chunk and W chunk needs one. The aim is to fit in those 32 and avoid spilling to the stack. That leaves the split: how many rows (R) and how many 4-float column chunks (K) per pass of the loop?

| | formula | R=4, K=4 | **R=16, K=1** | R=1, K=16 | R=8, K=2 |
|---|---|---|---|---|---|
| x loads | K | 4 | **1** | 16 | 2 |
| W loads | R × K | 16 | **16** | 16 | 16 |
| FMAs | R × K | 16 | **16** | 16 | 16 |
| **loads per FMA** | 1 + 1/R | 1.25 | **1.06** | 2 | 1.13 |
| x registers | K | 4 | **1** | 16 | 2 |
| sum registers | R × K | 16 | **16** | 16 | 16 |
| W registers | 1 to R × K | 1 to 16 | **1 to 16** | 1 to 16 | 1 to 16 |
| **total registers** | | 21 to 36 | **18 to 33** | 33 to 48 | 19 to 34 |

The sums and x chunks must all stay in registers at once: the sums are added to on every pass, and each x chunk is used by every FMA in its pass. W chunks can share registers. Each one is used by a single FMA and then never again, so the next load can reuse its register, and register renaming gives every load its own physical register behind the scenes, so the reuse doesn't make loads wait. In the compiled `matmul_simd`, clang reuses just a few registers for all 16 W loads.

That's why W registers are a range, and why only the lower bound has to fit in 32. R=1, K=16 already needs 32 registers for its sums and x chunks, with nothing left for W, so it would spill.

R=16, K=1 has the fewest loads per FMA and fits in 32 registers, so that's the split used.

### 2. The NEON kernel (`matmul_simd`)

16 rows of W in lockstep, 16 sums, one x chunk loaded per pass and reused for all 16 rows (register tiling):

```c
for (int i = 0; i < n; i += 4) {
    float32x4_t x_chunk = vld1q_f32(&x[i]);              // load x once
    sum_0  = vmlaq_f32(sum_0,  vld1q_f32(&w[(j+0)*n + i]),  x_chunk);
    sum_1  = vmlaq_f32(sum_1,  vld1q_f32(&w[(j+1)*n + i]),  x_chunk);
    ...
    sum_15 = vmlaq_f32(sum_15, vld1q_f32(&w[(j+15)*n + i]), x_chunk);
}
```

Tails: rows left over after the 16-row blocks go through a 1-row loop. Columns left over after the last full chunk of 4 are handled by masking x, so lanes past `n` become 0 (0 × w = 0). Both masks are built once, outside the loop.

Result: more than 2x the tok/s of -O2 and -O3, but about 2x slower than -Ofast.

### 3. Why it beats -O3 but not -Ofast

**-O3:** float adds aren't associative, so without `-ffast-math` the compiler isn't allowed to split `val += w * x` into several sums. It stays one scalar chain, every add waits for the last one, and it doesn't vectorise.

**-Ofast:** turns on `-ffast-math`, so it can reorder the adds into several sums and vectorise, which is close to what `matmul_simd` does by hand. That makes -Ofast the real baseline.

To see where the -Ofast gap comes from, the benchmark runs each kernel twice: W in cache (288 × 288, 330 KB, called over and over) and W from DRAM (32000 × 288, 37 MB, the real classifier). First result: `matmul_simd` is faster than -Ofast in cache (5,158 vs 7,375 ns) and slower from DRAM (18.4 vs 32.6 GB/s). The tok/s gap is a memory access problem.

### 4. Packing W (`matmul_simd_packed`)

In row-major order the 16 rows in one pass are 1,152 bytes apart, so each pass reads from 16 different places in memory: 16 streams of cache lines for the hardware prefetcher to follow. The -Ofast loop reads one row at a time, so one stream.

Packing rearranges W once at load time. For each block of 16 rows: 4 floats from row 0, then the same 4 columns from row 1, ... row 15, then the next 4 columns.

```
row-major: one pass reads 16 places, 1,152 B apart
  row 0:  [c0 c1 c2 c3][c4 c5 c6 c7] ...
  row 1:  [c0 c1 c2 c3][c4 c5 c6 c7] ...
  ...
  row 15: [c0 c1 c2 c3][c4 c5 c6 c7] ...

packed: one pass reads 256 contiguous bytes (two 128 B cache lines)
  [r0 c0-c3][r1 c0-c3] ... [r15 c0-c3][r0 c4-c7][r1 c4-c7] ...
```

In the loop, `j` (which block of 16 rows), `i` (which 4 columns) and `k` (which row in the block) mean the same as before. Only the address changes:

```
row-major: w  + (j + k) * n + i
packed:    wp + j * n + i * 16 + 4 * k
```

`i * 16` skips the earlier groups of 64 floats (16 rows × 4 columns each). The packed buffer is 128-byte aligned to match the M1's cache line.

Access from DRAM goes from 17 to 40 GB/s, and tok/s is level with -Ofast. Found out afterwards this is a standard technique, called packing.

To keep it simple, packing only runs when every matrix has rows divisible by 16 and columns divisible by 4, which is true for stories15M. Anything else falls back to `matmul_simd`, which handles the tails.

### 5. Mask cleanup

With W in cache, instruction count matters. The M1 front end takes in about 8 instructions per cycle across all types. Reading the assembly, the -Ofast loop is 10 instructions per pass for 4 FMAs (2.5 per FMA), while `matmul_simd` was 74 for 16 (4.6 per FMA): 14 `add`s rebuilding the 16 row addresses, plus mask instructions and loop control. 74 instructions take at least ~9 cycles per pass, when the 16 FMAs alone only need 4.

Only x needs masking, since 0 × w = 0, so the W masks come out of `matmul_simd`. `matmul_simd_packed` doesn't need masks at all, since it only runs when columns are divisible by 4.

| | instructions per pass, before | after | W in cache, before | after |
|---|---|---|---|---|
| `matmul_simd` | 74 | 56 | 6,307 ns | 5,797 ns (-8%) |
| `matmul_simd_packed` | 52 | 29 | 5,334 ns | 5,014 ns (-6%) |

No change from DRAM.

## Possible further optimisations

- **Software prefetch.** Not tried. Packing already turned 16 streams of cache lines (one per row in the lockstep) into 1, which is what the hardware prefetcher handles well.
- **Kernel fusion.** wq, wk and wv all multiply the same x, so q, k and v could be computed in one pass instead of three separate matmuls.
- **Quantisation.** `runq.c` already runs the model with int8 weights (a scale per group of weights), so a quarter of the bytes per weight to stream from DRAM. Its matmul could get the same NEON and packing treatment.
- **Threading.** Out of scope, just focused on one core.

## Open questions

- After the mask cleanup, `matmul_simd_packed` runs 44% fewer instructions per pass but is only 6% faster in cache. Something other than instruction count is limiting it.
- From DRAM, `matmul_simd_packed` is only slightly faster than -Ofast (40.0 vs 37.8 GB/s), while in cache it's 1.5x faster. My guess is both are capped by how fast one core can read from DRAM, but I haven't measured that limit.
