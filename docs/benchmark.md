# Benchmark

## Query + Insert Performance

These benchmarks compare the total time to execute a set number of 
queries and inserts across various Python spatial index libraries.
Quadtrees are the focus of the benchmark, but Rtrees are included for reference.


![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)

- Points: **250,000**, Queries: **500**
- Fastest total: **fastquadtree** at **0.120 s**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| **fastquadtree** | 0.031 | 0.089 | 0.120 | **14.64×** |
| Shapely STRtree | 0.179 | 0.100 | 0.279 | 6.29× |
| nontree-QuadTree | 0.595 | 0.605 | 1.200 | 1.46× |
| Rtree        | 0.961 | 0.300 | 1.261 | 1.39× |
| e-pyquadtree | 1.005 | 0.660 | 1.665 | 1.05× |
| PyQtree      | 1.492 | 0.263 | 1.755 | 1.00× |
| quads        | 1.407 | 0.484 | 1.890 | 0.93× |

### Benchmark Configuration
| Parameter | Value |
|---|---:|
| Bounds | (0, 0, 1000, 1000) |
| Max points per node | 128 |
| Max depth | 16 |
| Queries per experiment | 500 |

---------

## Native vs Shim

### Configuration
- Points: 500,000
- Queries: 500
- Repeats: 5

### Results

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| Native | 0.181 | 2.024 | 2.205 |
| Shim (no map) | 0.301 | 1.883 | 2.184 |
| Shim (track+objs) | 0.651 | 2.016 | 2.667 |

### Summary

Using the shim with object tracking increases build time by 3.604x and query time by 0.996x.
**Total slowdown = 1.210x.**

Adding the object map only impacts the build time, not the query time.

---------

## System Info
- **OS**: Windows 11 AMD64
- **Python**: CPython 3.12.2
- **CPU**: AMD Ryzen 7 3700X 8-Core Processor (16 threads)
- **Memory**: 31.9 GB
- **GPU**: NVIDIA GeForce RTX 5070 (11.9 GB)

## Running Benchmarks
To run the benchmarks yourself, first install the dependencies:

```bash
pip install -r benchmarks/requirements.txt
```

Then run:

```bash
python benchmarks/cross_library_bench.py
python benchmarks/benchmark_native_vs_shim.py 
```

Check the CLI arguments for the cross-library benchmark in `benchmarks/quadtree_bench/main.py`.

