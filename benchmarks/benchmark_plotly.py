import gc
import math
import random
import statistics as stats
from time import perf_counter as now
from typing import Callable, Dict, List, Tuple, Any, Optional
from tqdm import tqdm

# --- Built-ins (always available in this repo) ---
from pyquadtree.quadtree import QuadTree as EPyQuadTree          # e-pyquadtree
from pyqtree import Index as PyQTree                             # Pyqtree
from quadtree_rs import QuadTree as RustQuadTree                 # quadtree-rs

# --- Optional extras (import lazily in adapters) ---
# quads (pure Python quadtree)
# nontree (TreeMap with QuadTree mode)
# rtree (R-tree comparator)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------
# Config
# ----------------------
BOUNDS = (0, 0, 1000, 1000)
MAX_POINTS = 20          # node capacity where supported
MAX_DEPTH = 10           # depth cap for fairness where supported
N_QUERIES = 500          # per experiment
REPEATS = 3              # median over repeats

MAX_EXPERIMENT_POINT_SIZE = 500_000
EXPERIMENTS = [2, 4, 8, 16]
while EXPERIMENTS[-1] < MAX_EXPERIMENT_POINT_SIZE:
    EXPERIMENTS.append(int(EXPERIMENTS[-1] * 1.5))
if EXPERIMENTS[-1] > MAX_EXPERIMENT_POINT_SIZE:
    EXPERIMENTS[-1] = MAX_EXPERIMENT_POINT_SIZE

print(f"Experiments: {EXPERIMENTS}, {len(EXPERIMENTS)} steps")

RNG_SEED = 42

# Colors (keep originals)
C_EPY  = "#1f77b4"  # e-pyquadtree (blue)
C_RUST = "#ff7f0e"  # quadtree-rs (orange)
C_PYQT = "#2ca02c"  # PyQtree (green)
C_BASE = "#9467bd"  # Brute force (purple)

# Extra colors for new libs
C_QUADS   = "#8c564b"  # brown
C_NONTREE = "#17becf"  # cyan
C_RTREE   = "#e377c2"  # pink

# Names to show in plots
N_EPY   = "e-pyquadtree"
N_PYQT  = "PyQtree"
N_RUST  = "quadtree-rs"
N_BF    = "Brute force"
N_QUADS = "quads"
N_NTREE = "nontree-QuadTree"
N_RTREE = "Rtree"

# Registry order: show your three first, then extras alphabetically
PREFERRED_ORDER = [N_RUST, N_PYQT, N_EPY, N_BF, N_NTREE, N_QUADS, N_RTREE]

def median_or_nan(vals):
    cleaned = [x for x in vals if isinstance(x, (int, float)) and not math.isnan(x)]
    return stats.median(cleaned) if cleaned else math.nan


# ----------------------
# Data gen
# ----------------------
def gen_points(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    return [(rng.randint(0, 999), rng.randint(0, 999)) for _ in range(n)]

def gen_queries(m: int, rng: random.Random) -> List[Tuple[int, int, int, int]]:
    qs = []
    for _ in range(m):
        x = rng.randint(0, 1000)
        y = rng.randint(0, 1000)
        w = rng.randint(0, 1000 - x)
        h = rng.randint(0, 1000 - y)
        qs.append((x, y, x + w, y + h))
    return qs

# ----------------------
# Adapter layer
# ----------------------
class Engine:
    """
    Adapter interface for each index.
    build(points) -> opaque tree
    query(tree, queries) -> iterate results or just perform lookups
    """
    def __init__(self, name: str, color: str,
                 build_fn: Callable[[List[Tuple[int,int]]], Any],
                 query_fn: Callable[[Any, List[Tuple[int,int,int,int]]], None]):
        self.name = name
        self.color = color
        self._build = build_fn
        self._query = query_fn

    def build(self, pts):
        return self._build(pts)

    def query(self, tree, qs):
        return self._query(tree, qs)

def _engine_e_pyquadtree() -> Engine:
    def build(points):
        qt = EPyQuadTree(BOUNDS, MAX_POINTS, MAX_DEPTH)
        for p in points:
            qt.add(None, p)
        return qt
    def query(qt, queries):
        for q in queries:
            _ = qt.query(q)
    return Engine(N_EPY, C_EPY, build, query)

def _engine_pyqtree() -> Engine:
    def build(points):
        qt = PyQTree(bbox=BOUNDS, max_items=MAX_POINTS, max_depth=MAX_DEPTH)
        for x, y in points:
            qt.insert(None, bbox=(x, y, x + 1, y + 1))
        return qt
    def query(qt, queries):
        for q in queries:
            _ = list(qt.intersect(q))
    return Engine(N_PYQT, C_PYQT, build, query)

def _engine_quadtree_rs() -> Engine:
    def build(points):
        qt = RustQuadTree(BOUNDS, MAX_POINTS, max_depth=MAX_DEPTH)
        for p in points:
            qt.insert(p)
        return qt
    def query(qt, queries):
        for q in queries:
            _ = qt.query(q)
    return Engine(N_RUST, C_RUST, build, query)

def _engine_quads() -> Optional[Engine]:
    try:
        import quads as qd
    except Exception:
        return None
    def build(points):
        (xmin, ymin, xmax, ymax) = BOUNDS
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w  = xmax - xmin
        h  = ymax - ymin
        tree = qd.QuadTree((cx, cy), w, h, capacity=MAX_POINTS)
        for p in points:
            tree.insert(p)  # accepts tuple or qd.Point
        return tree
    def query(tree, queries):
        import quads as qd
        for (xmin, ymin, xmax, ymax) in queries:
            bb = qd.BoundingBox(min_x=xmin, min_y=ymin, max_x=xmax, max_y=ymax)
            _ = tree.within_bb(bb)
    return Engine(N_QUADS, C_QUADS, build, query)

def _engine_nontree() -> Optional[Engine]:
    try:
        from nontree.TreeMap import TreeMap
    except Exception:
        return None
    def build(points):
        (xmin, ymin, xmax, ymax) = BOUNDS
        w = xmax - xmin
        h = ymax - ymin
        tm = TreeMap((xmin, ymin, w, h), mode=4, bucket=MAX_POINTS, lvl=MAX_DEPTH)  # 4 => QuadTree
        # store a tiny payload to match API; value is irrelevant
        for p in points:
            tm[p] = 1
        return tm
    def query(tm: TreeMap, queries):
        for (xmin, ymin, xmax, ymax) in queries:
            _ = tm.get_rect((xmin, ymin, xmax - xmin, ymax - ymin))
    return Engine(N_NTREE, C_NONTREE, build, query)

def _engine_rtree() -> Optional[Engine]:
    try:
        from rtree import index as rindex
    except Exception:
        return None
    
    def build(points):
        # Tune a few sane defaults
        p = rindex.Property()
        p.dimension = 2
        p.variant = rindex.RT_Star
        p.leaf_capacity = 1000
        p.index_capacity = 1000
        p.fill_factor = 0.9

        # Bulk stream loading: pass a generator of (id, bbox, obj)
        def stream():
            for i, (x, y) in enumerate(points):
                yield (i, (x, y, x + 1, y + 1), None)

        # Memory storage is default when no filename is given
        idx = rindex.Index(stream(), properties=p)
        return idx
    
    def query(idx, queries):
        for q in queries:
            _ = list(idx.intersection(q))
    return Engine(N_RTREE, C_RTREE, build, query)

def get_engines() -> Dict[str, Engine]:
    base = {
        N_EPY:  _engine_e_pyquadtree(),
        N_PYQT: _engine_pyqtree(),
        N_RUST: _engine_quadtree_rs(),
    }
    # Optional ones, only include if import succeeded
    for maker in (_engine_quads, _engine_nontree, _engine_rtree):
        eng = maker()
        if eng is not None:
            base[eng.name] = eng
    return base

# ----------------------
# One pass for a single engine
# ----------------------
def bench_engine_once(engine: Engine,
                      points: List[Tuple[int,int]],
                      queries: List[Tuple[int,int,int,int]]) -> Tuple[float, float]:
    # separate build vs query
    t0 = now()
    tree = engine.build(points)
    t_build = now() - t0

    t0 = now()
    engine.query(tree, queries)
    t_query = now() - t0
    return t_build, t_query

# ----------------------
# Main loop
# ----------------------
def run_bench():
    rng = random.Random(RNG_SEED)
    engines = get_engines()

    # Warmup on a small set to JIT caches, etc.
    _ = [bench_engine_once(e, gen_points(2_000, rng), gen_queries(N_QUERIES, rng))
         for e in engines.values()]

    # results dicts keyed by engine name
    total      : Dict[str, List[float]] = {k: [] for k in engines}
    build      : Dict[str, List[float]] = {k: [] for k in engines if k != N_BF}
    query      : Dict[str, List[float]] = {k: [] for k in engines}
    insert_rate: Dict[str, List[float]] = {k: [] for k in engines if k != N_BF}
    query_rate : Dict[str, List[float]] = {k: [] for k in engines}

    # add brute-force query as a special entry
    query[N_BF] = []
    total[N_BF] = []
    query_rate[N_BF] = []
    iterator = tqdm(EXPERIMENTS, desc="Experiments", unit="points")
    for n in iterator:
        iterator.set_postfix({"points": n})
        r_local = random.Random(10_000 + n)
        pts = gen_points(n, r_local)
        qs  = gen_queries(N_QUERIES, r_local)

        # collect repeats for medians
        per_engine_times = {name: {"b": [], "q": []} for name in engines}
        bf_q = []

        for _ in range(REPEATS):
            gc.disable()
            for name, eng in engines.items():
                try:
                    eb, eq = bench_engine_once(eng, pts, qs)
                except Exception:
                    # mark as failed in this repeat; skip contribution
                    eb, eq = math.nan, math.nan
                per_engine_times[name]["b"].append(eb)
                per_engine_times[name]["q"].append(eq)

            # brute-force query-only
            t0 = now()
            for q in qs:
                _ = [p for p in pts if q[0] <= p[0] <= q[2] and q[1] <= p[1] <= q[3]]
            bf_q.append(now() - t0)
            gc.enable()

        # reduce by median
        for name in engines:
            b_med = median_or_nan(per_engine_times[name]["b"])
            q_med = median_or_nan(per_engine_times[name]["q"])

            build[name].append(b_med)
            query[name].append(q_med)
            total[name].append(b_med + q_med)
            insert_rate[name].append((n / b_med) if b_med and b_med > 0 else 0.0)
            query_rate[name].append((N_QUERIES / q_med) if q_med and q_med > 0 else 0.0)

        bf_med = stats.median(bf_q)
        query[N_BF].append(bf_med)
        total[N_BF].append(bf_med)
        query_rate[N_BF].append((N_QUERIES / bf_med) if bf_med and bf_med > 0 else 0.0)

    return {"total": total, "build": build, "query": query,
            "insert_rate": insert_rate, "query_rate": query_rate, "engines": engines}

# ----------------------
# Figures
# ----------------------
def _sorted_names(names: List[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(PREFERRED_ORDER)}
    return sorted(names, key=lambda n: order_map.get(n, 10_000 + hash(n) % 1000))

def make_figures(results):
    engines = results["engines"]
    total = results["total"]; build = results["build"]; query = results["query"]
    insert_rate = results["insert_rate"]; query_rate = results["query_rate"]

    # Time panels
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Total time", "Build time", "Query time"),
        horizontal_spacing=0.08
    )

    def add_time_traces(y_map, col):
        show = (col == 1)  # only show legend for the first column
        for name in _sorted_names(list(y_map.keys())):
            if name not in engines and name != N_BF:
                continue
            color = (engines[name].color if name in engines else C_BASE)
            fig.add_trace(go.Scatter(
                x=EXPERIMENTS, y=y_map[name], name=name,
                legendgroup=name, showlegend=show,
                line=dict(color=color, width=3)
            ), row=1, col=col)

    add_time_traces(total, 1)
    add_time_traces(build, 2)
    add_time_traces(query, 3)

    for c in (1, 2, 3):
        fig.update_xaxes(title_text="Number of points", row=1, col=c)
        fig.update_yaxes(title_text="Time (s)", row=1, col=c)

    fig.update_layout(
        title=f"Tree build and query benchmarks (Max Depth {MAX_DEPTH}, Capacity {MAX_POINTS}, {REPEATS}x median, {N_QUERIES} queries)",
        template="plotly_dark",
        legend=dict(orientation="v", traceorder="normal", xanchor="left", x=0, yanchor="top", y=1),
        margin=dict(l=40, r=20, t=80, b=40),
        height=520,
    )

    # Throughput panels
    fig_rate = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Insert rate (points/sec)", "Query rate (queries/sec)"),
        horizontal_spacing=0.12
    )
    fig_rate.update_yaxes(type="log", row=1, col=2)

    # Insert traces (left) and Query traces (right)
    names_for_rate = _sorted_names(list(query_rate.keys()))
    for name in names_for_rate:
        color = C_BASE if name == N_BF else engines[name].color
        if name in insert_rate:
            fig_rate.add_trace(go.Scatter(
                x=EXPERIMENTS, y=insert_rate[name],
                name=name, legendgroup=name, showlegend=False,
                line=dict(color=color, width=3)
            ), row=1, col=1)
        fig_rate.add_trace(go.Scatter(
            x=EXPERIMENTS, y=query_rate[name],
            name=name, legendgroup=name, showlegend=True,
            line=dict(color=color, width=3)
        ), row=1, col=2)

    fig_rate.update_xaxes(title_text="Number of points", row=1, col=1)
    fig_rate.update_xaxes(title_text="Number of points", row=1, col=2)
    fig_rate.update_yaxes(title_text="Ops/sec", row=1, col=1)
    fig_rate.update_yaxes(title_text="Ops/sec", row=1, col=2)
    fig_rate.update_layout(
        title="Throughput",
        template="plotly_dark",
        legend=dict(orientation="v", traceorder="normal", xanchor="left", x=-8),
        margin=dict(l=40, r=20, t=80, b=40),
        height=480,
    )

    return fig, fig_rate

# ----------------------
# Markdown summary (PyQtree baseline)
# ----------------------
def print_markdown_summary(results):
    total = results["total"]; build = results["build"]; query = results["query"]
    engines = results["engines"]
    i = len(EXPERIMENTS) - 1
    fmt = lambda x: f"{x:.3f}" if x is not None and not math.isnan(x) else "nan"

    print("\n### Summary (largest dataset, PyQtree baseline)")
    print(f"- Points: **{EXPERIMENTS[i]:,}**, Queries: **{N_QUERIES}**")
    print("--------------------")

    # Baseline times
    py_total = total[N_PYQT][i]
    # Rank all by total time
    ranked = sorted(total.keys(), key=lambda n: total[n][i])
    best = ranked[0]
    print(f"- Fastest total: **{best}** at **{fmt(total[best][i])} s**")
    print(f"- PyQtree total: **{fmt(py_total)} s**")
    if N_RUST in total:
        print(f"- quadtree-rs total: **{fmt(total[N_RUST][i])} s**")
    if N_EPY in total:
        print(f"- e-pyquadtree total: **{fmt(total[N_EPY][i])} s**")
    if N_BF in total:
        print(f"- Brute force total: **{fmt(total[N_BF][i])} s**")
    print("--------------------")

    # Table header
    print("\n| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |")
    print("|---|---:|---:|---:|---:|")

    def rel_speed(name: str) -> str:
        t = total[name][i]
        return f"{(py_total / t):.2f}×" if t and t > 0 else "n/a"

    row_names = _sorted_names(list(total.keys()))
    for name in row_names:
        b = build.get(name, [math.nan])[i] if name in build else math.nan
        q = query[name][i]
        t = total[name][i]
        print(f"| {name:12} | {fmt(b)} | {fmt(q)} | {fmt(t)} | {rel_speed(name)} |")
    print("")

# ----------------------
# Save
# ----------------------
def save_figures(fig_time, fig_rate, out_prefix="quadtree_bench"):
    try:
        fig_time.write_image(f"assets/{out_prefix}_time.png", scale=2, width=1200, height=520)
        fig_rate.write_image(f"assets/{out_prefix}_throughput.png", scale=2, width=1200, height=480)
        print("Saved PNGs via kaleido.")
    except Exception as e:
        print("Skipped PNG export. Install kaleido to save PNGs:", e)

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    results = run_bench()
    fig_time, fig_rate = make_figures(results)
    print_markdown_summary(results)
    save_figures(fig_time, fig_rate)
    # fig_time.show(); fig_rate.show()
