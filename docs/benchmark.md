# Benchmark

## Query + Insert Performance

These benchmarks compare the total time to execute a set number of 
queries and inserts across various Python spatial index libraries.
Quadtrees are the focus of the benchmark, but Rtrees are included for reference.


![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **250,000**, Queries: **500**
- Fastest total: **fastquadtree** at **0.030 s**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| fastquadtree | 0.023 | 0.007 | 0.030 | 41.31× |
| Shapely STRtree | 0.094 | 0.049 | 0.143 | 8.67× |
| nontree-QuadTree | 0.448 | 0.475 | 0.924 | 1.34× |
| Rtree        | 0.801 | 0.225 | 1.025 | 1.21× |
| e-pyquadtree | 0.666 | 0.451 | 1.117 | 1.11× |
| PyQtree      | 1.066 | 0.175 | 1.241 | 1.00× |
| quads        | 0.969 | 0.330 | 1.299 | 0.96× |

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
- Repeats: 5

### Results

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| Native | 0.134 | 1.873 | 2.007 |
| Shim (no tracking) | 0.245 | 1.815 | 2.060 |
| Shim (tracking) | 0.714 | 2.062 | 2.776 |

### Summary

Using the shim with object tracking increases build time by 5.345x and query time by 1.101x.
**Total slowdown = 1.383x.**

Adding the object map tends to impact the build time more than query time.

-----------

## pyqtree drop-in shim performance gains

### Configuration
- Points: 500,000
- Queries: 500
- Repeats: 5

### Results

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| pyqtree (fastquadtree) | 0.443 | 2.007 | 2.450 |
| pyqtree (original) | 3.233 | 12.857 | 16.091 |

### Summary

If you directly replace pyqtree with the drop-in `fastquadtree.pyqtree.Index` shim, you get a build time of 0.443s and query time of 2.007s.
This is a **total speedup of 6.567x** compared to the original pyqtree and requires no code changes.

---------

## NumPy Bulk Insert vs Python List Insert
### Configuration

- Points: 500,000
- Repeats: 5
- Dtype: float32
- Track objects: False

### Results (median of repeats)

| Variant | Build time |
|---|---:|
| NumPy array direct | 42.8 ms |
| Python list insert only | 51.1 ms |
| Python list including conversion | 540.2 ms |

Key:  

- *NumPy array direct*: Using the `insert_many` method with a NumPy array of shape (N, 2).  
- *Python list insert only*: Using the `insert_many` method with a Python list of tuples.  
- *Python list including conversion*: Time taken to convert a NumPy array to a Python list of tuples, then inserting.  

### Summary
If your data is already in a NumPy array, using the `insert_many` method directly with the array is significantly faster than converting to a Python list first.

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

