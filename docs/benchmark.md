# Benchmark

## Query + Insert Performance

These benchmarks compare the total time to execute a set number of 
queries and inserts across various Python spatial index libraries.
Quadtrees are the focus of the benchmark, but Rtrees are included for reference.


![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **500,000**, Queries: **500**
- Fastest total: **fastquadtree** at **0.089 s**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| fastquadtree | 0.063 | 0.026 | 0.089 | 48.99× |
| Shapely STRtree | 0.314 | 0.178 | 0.492 | 8.90× |
| fastquadtree (obj tracking) | 0.388 | 0.244 | 0.632 | 6.93× |
| nontree-QuadTree | 1.180 | 1.287 | 2.467 | 1.77× |
| Rtree        | 1.905 | 0.622 | 2.527 | 1.73× |
| e-pyquadtree | 2.083 | 1.479 | 3.562 | 1.23× |
| quads        | 3.058 | 1.140 | 4.198 | 1.04× |
| PyQtree      | 3.775 | 0.603 | 4.378 | 1.00× |

#### Benchmark Configuration
| Parameter | Value |
|---|---:|
| Bounds | (0, 0, 1000, 1000) |
| Max points per node | 128 |
| Max depth | 16 |
| Queries per experiment | 500 |

> Fastquadtree is using query_np to return Numpy arrays rather than typical Python objects

---------

## Native vs Shim

### Configuration
- Points: 500,000
- Queries: 500
- Repeats: 3

### Results

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| Native | 0.141 | 1.858 | 1.999 |
| QuadTree (no objects) | 0.235 | 1.920 | 2.155 |
| QuadTreeObjects | 0.927 | 2.052 | 2.979 |
| QuadTree (numpy, no objects) | 0.046 | 0.252 | 0.297 |

### Summary

- The Python shim (QuadTree) is 1.078x slower than the native engine due to Python overhead.

- NumPy points are the fastest path: build is **5.157x faster** than the list path and queries are **7.627x faster**,
  for a **7.249x** total speedup vs the list path.

- QuadTreeObjects adds object association overhead. Build time increases significantly, query time is moderately slower.

## pyqtree drop-in shim performance gains

### Configuration
- Points: 500,000
- Queries: 500
- Repeats: 3

### Results

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| pyqtree (fastquadtree) | 0.485 | 2.086 | 2.571 |
| pyqtree (original) | 3.463 | 13.209 | 16.672 |

### Summary

If you directly replace pyqtree with the drop-in `fastquadtree.pyqtree.Index` shim, you get a build time of 0.485s and query time of 2.086s.
This is a **total speedup of 6.486x** compared to the original pyqtree and requires no code changes.

---------

## NumPy Bulk Insert vs Python List Insert
### Configuration

- Points: 500,000
- Repeats: 5
- Dtype: float32

Results (median of repeats)

| Variant | Build time |
|---|---:|
| NumPy array direct | 42.3 ms |
| Python list insert only | 58.2 ms |
| Python list including conversion | 573.4 ms |

Key:  

- *NumPy array direct*: Using the `insert_many_np` method with a NumPy array of shape (N, 2).  
- *Python list insert only*: Using the `insert_many` method with a Python list of tuples.  
- *Python list including conversion*: Time taken to convert a NumPy array to a Python list of tuples, then inserting.  

### Summary
If your data is already in a NumPy array, using the `insert_many_np` method directly with the array is significantly faster than converting to a Python list first.

---------

## Serialization vs Rebuild

### Configuration
- Points: 1,000,000
- Capacity: 64
- Max depth: 10
- Repeats: 7

### Results

| Variant | Mean (s) | Stdev (s) |
|---|---:|---:|
| Serialize to bytes | 0.021356 | 0.000937 |
| Rebuild from points | 0.106783 | 0.011430 |
| Rebuild from bytes | 0.021754 | 0.001687 |
| Rebuild from file | 0.024887 | 0.001846 |

### Summary

- Rebuild from bytes is **4.908747x** faster than reinserting points.
- Rebuild from file is **4.290712x** faster than reinserting points.
- Serialized blob size is **13,770,328 bytes**.

----------------

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
python benchmarks/benchmark_np_vs_list.py 
python benchmarks/benchmark_serialization_vs_rebuild.py
```

Check the CLI arguments for the cross-library benchmark in `benchmarks/quadtree_bench/main.py`.

