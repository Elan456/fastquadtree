# benchmarks/bench_plotly.py
import gc
import random
import statistics as stats
from time import perf_counter as now
from tqdm import tqdm 

from pyquadtree.quadtree import QuadTree as EPyQuadTree
from pyqtree import Index as PyQTree
from quadtree_rs import QuadTree as RustQuadTree

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------
# Config
# ----------------------
BOUNDS = (0, 0, 1000, 1000)
MAX_POINTS = 20          # node capacity
MAX_DEPTH = 10           # depth cap for fairness
N_QUERIES = 500          # per experiment
REPEATS = 3              # median over repeats
EXPERIMENTS = list(range(0, 120_000, 20_000))
RNG_SEED = 42

# Colors
C_EPY  = "#1f77b4"  # e-pyquadtree (blue)
C_RUST = "#ff7f0e"  # quadtree-rs (orange)
C_PYQT = "#2ca02c"  # PyQtree (green, baseline)
C_BASE = "#9467bd"  # Brute force (purple)

# ----------------------
# Data gen
# ----------------------
def gen_points(n, rng):
    return [(rng.randint(0, 1000), rng.randint(0, 1000)) for _ in range(n)]

def gen_queries(m, rng):
    qs = []
    for _ in range(m):
        x = rng.randint(0, 1000)
        y = rng.randint(0, 1000)
        w = rng.randint(0, 1000 - x)
        h = rng.randint(0, 1000 - y)
        qs.append((x, y, x + w, y + h))
    return qs

# ----------------------
# One pass, split build vs query
# ----------------------
def bench_once(points, queries, max_points, max_depth):
    # e-pyquadtree (points)
    t0 = now()
    qt_e = EPyQuadTree(BOUNDS, max_points, max_depth)
    for p in points:
        qt_e.add(None, p)
    t_e_build = now() - t0

    t0 = now()
    for q in queries:
        _ = qt_e.query(q)
    t_e_query = now() - t0

    # PyQtree (AABB)
    t0 = now()
    qt_py = PyQTree(bbox=BOUNDS, max_items=max_points, max_depth=max_depth)
    for x, y in points:
        qt_py.insert(None, bbox=(x, y, x + 1, y + 1))
    t_py_build = now() - t0

    t0 = now()
    for q in queries:
        _ = list(qt_py.intersect(q))
    t_py_query = now() - t0

    # Rust (points)
    t0 = now()
    qt_rs = RustQuadTree(BOUNDS, max_points, max_depth=max_depth)
    for i, p in enumerate(points):
        qt_rs.insert(i, p)
    t_rs_build = now() - t0

    t0 = now()
    for q in queries:
        _ = qt_rs.query(q)
    t_rs_query = now() - t0

    # Brute force query only
    t0 = now()
    for q in queries:
        _ = [p for p in points if q[0] <= p[0] <= q[2] and q[1] <= p[1] <= q[3]]
    t_base_query = now() - t0

    return (
        (t_e_build, t_e_query),
        (t_py_build, t_py_query),
        (t_rs_build, t_rs_query),
        (0.0, t_base_query),
    )

# ----------------------
# Main loop
# ----------------------
def run_bench():
    rng = random.Random(RNG_SEED)
    _ = bench_once(gen_points(2_000, rng), gen_queries(N_QUERIES, rng), MAX_POINTS, MAX_DEPTH)  # warmup

    total = {"e-pyquadtree": [], "PyQtree": [], "quadtree-rs": [], "Brute force": []}
    build = {"e-pyquadtree": [], "PyQtree": [], "quadtree-rs": []}
    query = {"e-pyquadtree": [], "PyQtree": [], "quadtree-rs": [], "Brute force": []}
    insert_rate = {"e-pyquadtree": [], "PyQtree": [], "quadtree-rs": []}
    query_rate = {"e-pyquadtree": [], "PyQtree": [], "quadtree-rs": [], "Brute force": []}

    for n in tqdm(EXPERIMENTS):
        r_local = random.Random(10_000 + n)
        pts = gen_points(n, r_local)
        qs = gen_queries(N_QUERIES, r_local)

        e_b, e_q = [], []
        py_b, py_q = [], []
        rs_b, rs_q = [], []
        bf_q = []

        for _ in range(REPEATS):
            gc.disable()
            (eb, eq), (pb, pq), (rb, rq), (_, bq) = bench_once(pts, qs, MAX_POINTS, MAX_DEPTH)
            gc.enable()
            e_b.append(eb); e_q.append(eq)
            py_b.append(pb); py_q.append(pq)
            rs_b.append(rb); rs_q.append(rq)
            bf_q.append(bq)

        e_bm, e_qm = stats.median(e_b), stats.median(e_q)
        py_bm, py_qm = stats.median(py_b), stats.median(py_q)
        rs_bm, rs_qm = stats.median(rs_b), stats.median(rs_q)
        bf_qm = stats.median(bf_q)

        build["e-pyquadtree"].append(e_bm)
        build["PyQtree"].append(py_bm)
        build["quadtree-rs"].append(rs_bm)

        query["e-pyquadtree"].append(e_qm)
        query["PyQtree"].append(py_qm)
        query["quadtree-rs"].append(rs_qm)
        query["Brute force"].append(bf_qm)

        total["e-pyquadtree"].append(e_bm + e_qm)
        total["PyQtree"].append(py_bm + py_qm)
        total["quadtree-rs"].append(rs_bm + rs_qm)
        total["Brute force"].append(bf_qm)

        insert_rate["e-pyquadtree"].append((n / e_bm) if e_bm > 0 else 0)
        insert_rate["PyQtree"].append((n / py_bm) if py_bm > 0 else 0)
        insert_rate["quadtree-rs"].append((n / rs_bm) if rs_bm > 0 else 0)

        query_rate["e-pyquadtree"].append(N_QUERIES / e_qm if e_qm > 0 else 0)
        query_rate["PyQtree"].append(N_QUERIES / py_qm if py_qm > 0 else 0)
        query_rate["quadtree-rs"].append(N_QUERIES / rs_qm if rs_qm > 0 else 0)
        query_rate["Brute force"].append(N_QUERIES / bf_qm if bf_qm > 0 else 0)

    return {"total": total, "build": build, "query": query,
            "insert_rate": insert_rate, "query_rate": query_rate}

# ----------------------
# Figures
# ----------------------
def make_figures(results):
    total = results["total"]; build = results["build"]; query = results["query"]
    insert_rate = results["insert_rate"]; query_rate = results["query_rate"]

    # 1) Time panels
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Total time", "Build time", "Query time"),
        horizontal_spacing=0.08
    )

    def add_time_traces(y_map, col):
        show = (col == 1)  # only show legend for the first column
        fig.add_trace(go.Scatter(
            x=EXPERIMENTS, y=y_map["PyQtree"], name="PyQtree",
            legendgroup="PyQtree", showlegend=show,
            line=dict(color=C_PYQT, width=3)
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=EXPERIMENTS, y=y_map["e-pyquadtree"], name="e-pyquadtree",
            legendgroup="e-pyquadtree", showlegend=show,
            line=dict(color=C_EPY, width=3)
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=EXPERIMENTS, y=y_map["quadtree-rs"], name="quadtree-rs",
            legendgroup="quadtree-rs", showlegend=show,
            line=dict(color=C_RUST, width=3)
        ), row=1, col=col)
        if "Brute force" in y_map:
            fig.add_trace(go.Scatter(
                x=EXPERIMENTS, y=y_map["Brute force"], name="Brute force",
                legendgroup="Brute force", showlegend=show,
                line=dict(color=C_BASE, width=3)
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

    # 2) Throughput panels
    fig_rate = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Insert rate (points/sec)", "Query rate (queries/sec)"),
        horizontal_spacing=0.12
    )
    fig_rate.update_yaxes(type="log", row=1, col=2)

    for name, color in [("PyQtree", C_PYQT), ("e-pyquadtree", C_EPY), ("quadtree-rs", C_RUST), ("Brute force", C_BASE)]:
        # Insert (left subplot) => shows legend
        if name in insert_rate:
            fig_rate.add_trace(go.Scatter(
                x=EXPERIMENTS, y=insert_rate[name],
                name=name, legendgroup=name, showlegend=False,
                line=dict(color=color, width=3)
            ), row=1, col=1)

        # Query (right subplot) => hidden from legend but in same group
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
        legend=dict(orientation="v", traceorder="normal", xanchor="left", x=0, yanchor="top", y=1),
        margin=dict(l=40, r=20, t=80, b=40),
        height=480,
    )

    return fig, fig_rate

# ----------------------
# Markdown summary (PyQtree baseline)
# ----------------------
def print_markdown_summary(results):
    total = results["total"]; build = results["build"]; query = results["query"]
    i = len(EXPERIMENTS) - 1
    fmt = lambda x: f"{x:.3f}"

    py = total["PyQtree"][i]
    e  = total["e-pyquadtree"][i]
    rs = total["quadtree-rs"][i]
    bf = total["Brute force"][i]

    print("\n### Summary (largest dataset, PyQtree baseline)")
    print(f"- Points: **{EXPERIMENTS[i]:,}**, Queries: **{N_QUERIES}**")
    print(f"--------------------")
    print(f"- Brute force total: **{fmt(bf)} s**")
    print(f"- e-pyquadtree total: **{fmt(e)} s**")
    print(f"- PyQtree total: **{fmt(py)} s**")
    print(f"- quadtree-rs total: **{fmt(rs)} s**")
    print(f"--------------------")
    print("\n| Library | Build (s) | Query (s) | Total (s) |")
    print("|---|---:|---:|---:|")
    print(f"| Brute force  | - | {fmt(query['Brute force'][i])} | {fmt(total['Brute force'][i])} |")
    print(f"| e-pyquadtree | {fmt(build['e-pyquadtree'][i])} | {fmt(query['e-pyquadtree'][i])} | {fmt(total['e-pyquadtree'][i])} |")
    print(f"| PyQtree      | {fmt(build['PyQtree'][i])} | {fmt(query['PyQtree'][i])} | {fmt(total['PyQtree'][i])} |")
    print(f"| quadtree-rs  | {fmt(build['quadtree-rs'][i])} | {fmt(query['quadtree-rs'][i])} | {fmt(total['quadtree-rs'][i])} |")
    print("")

# ----------------------
# Save
# ----------------------
def save_figures(fig_time, fig_rate, out_prefix="quadtree_bench"):
    # Static PNGs for README (pip install kaleido)
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
    fig_time, fig_rate= make_figures(results)
    print_markdown_summary(results)
    save_figures(fig_time, fig_rate)
    # fig_time.show(); fig_rate.show(); fig_speed.show()
