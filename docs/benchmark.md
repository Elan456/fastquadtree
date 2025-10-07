# Benchmark

## Query + Insert Performance

These benchmarks compare the total time to execute a set number of 
queries and inserts across various Python spatial index libraries.
Quadtrees are the focus of the benchmark, but Rtrees are included for reference.


![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **250,000**, Queries: **500**
--------------------
- Fastest total: **fastquadtree** at **0.120 s**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| fastquadtree | 0.031 | 0.089 | 0.120 | 14.64× |
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

## Native vs Shim Benchmark

**Setup**
- Points: 500,000
- Queries: 500
- Repeats: 5

**Timing (seconds)**

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| Native | 0.483 | 4.380 | 4.863 |
| Shim (no map) | 0.668 | 4.167 | 4.835 |
| Shim (track+objs) | 1.153 | 4.458 | 5.610 |

**Overhead vs Native**

- No map: build 1.38x, query 0.95x, total 0.99x  
- Track + objs: build 2.39x, query 1.02x, total 1.15x

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

