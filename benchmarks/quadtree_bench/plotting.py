"""
Plotting and visualization for quadtree benchmarks.

This module handles creation of performance charts, graphs, and visualizations
for benchmark results analysis.
"""

from typing import Any, Dict, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlotManager:
    """Manages creation and export of benchmark visualization plots."""

    def __init__(self, results: Dict[str, Any]):
        """Initialize with benchmark results."""
        self.results = results
        self.config = results["config"]
        self.engines = results["engines"]

    def create_time_plots(self) -> go.Figure:
        """Create time-based performance plots (total, build, query)."""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Total time", "Build time", "Query time"),
            horizontal_spacing=0.08,
        )

        def add_time_traces(y_data: Dict[str, list], col: int):
            """Add traces for a specific time metric."""
            show_legend = col == 1  # Only show legend for first column
            for name in list(y_data.keys()):
                # Determine color
                color = self.engines[name].color

                fig.add_trace(
                    go.Scatter(
                        x=self.config.experiments,
                        y=y_data[name],
                        name=name,
                        legendgroup=name,
                        showlegend=show_legend,
                        line={"color": color, "width": 3},
                    ),
                    row=1,
                    col=col,
                )

        # Add traces for each time metric
        add_time_traces(self.results["total"], 1)
        add_time_traces(self.results["build"], 2)
        add_time_traces(self.results["query"], 3)

        # Update axes
        for col in (1, 2, 3):
            fig.update_xaxes(title_text="Number of points", row=1, col=col)
            fig.update_yaxes(title_text="Time (s)", row=1, col=col)

        # Update layout
        fig.update_layout(
            title=(
                f"Tree build and query benchmarks "
                f"(Max Depth {self.config.max_depth}, "
                f"Capacity {self.config.max_points}, "
                f"{self.config.repeats}x median, "
                f"{self.config.n_queries} queries)"
            ),
            template="plotly_dark",
            legend={
                "orientation": "v",
                "traceorder": "normal",
                "xanchor": "left",
                "x": 0,
                "yanchor": "top",
                "y": 1,
            },
            margin={"l": 40, "r": 20, "t": 80, "b": 40},
            height=520,
        )

        return fig

    def create_throughput_plots(self) -> go.Figure:
        """Create throughput performance plots (insert rate, query rate)."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Insert rate (points/sec)", "Query rate (queries/sec)"),
            horizontal_spacing=0.12,
        )

        # Set logarithmic scale for query rate
        fig.update_yaxes(type="log", row=1, col=2)

        for name in list(self.results["query_rate"].keys()):
            # Determine color
            color = self.engines[name].color

            # Add insert rate trace (if available)
            if name in self.results["insert_rate"]:
                fig.add_trace(
                    go.Scatter(
                        x=self.config.experiments,
                        y=self.results["insert_rate"][name],
                        name=name,
                        legendgroup=name,
                        showlegend=False,
                        line={"color": color, "width": 3},
                    ),
                    row=1,
                    col=1,
                )

            # Add query rate trace
            fig.add_trace(
                go.Scatter(
                    x=self.config.experiments,
                    y=self.results["query_rate"][name],
                    name=name,
                    legendgroup=name,
                    showlegend=True,
                    line={"color": color, "width": 3},
                ),
                row=1,
                col=2,
            )

        # Update axes
        fig.update_xaxes(title_text="Number of points", row=1, col=1)
        fig.update_xaxes(title_text="Number of points", row=1, col=2)
        fig.update_yaxes(title_text="Ops/sec", row=1, col=1)
        fig.update_yaxes(title_text="Ops/sec", row=1, col=2)

        # Update layout
        fig.update_layout(
            title="Throughput",
            template="plotly_dark",
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.08,  # above the subplots
                "yanchor": "bottom",
            },
            margin={"l": 60, "r": 40, "t": 120, "b": 40},
            height=480,
        )

        return fig

    def create_all_plots(self) -> Tuple[go.Figure, go.Figure]:
        """Create all benchmark plots."""
        time_fig = self.create_time_plots()
        throughput_fig = self.create_throughput_plots()
        return time_fig, throughput_fig

    def save_plots(
        self,
        time_fig: go.Figure,
        throughput_fig: go.Figure,
        output_prefix: str = "quadtree_bench",
        output_dir: str = "assets",
    ) -> None:
        """
        Save plots as PNG images.

        Args:
            time_fig: Time performance figure
            throughput_fig: Throughput performance figure
            output_prefix: Prefix for output filenames
            output_dir: Directory to save images
        """
        try:
            time_fig.write_image(
                f"{output_dir}/{output_prefix}_time.png",
                scale=2,
                width=1200,
                height=520,
            )
            throughput_fig.write_image(
                f"{output_dir}/{output_prefix}_throughput.png",
                scale=2,
                width=1200,
                height=480,
            )
            print(f"Saved PNG images to {output_dir}/ with prefix '{output_prefix}'")
        except Exception as e:  # noqa: BLE001
            print(
                f"Failed to save PNG images. Install kaleido to enable PNG export: {e}"
            )

    def show_plots(self, time_fig: go.Figure, throughput_fig: go.Figure) -> None:
        """Display plots in browser."""
        time_fig.show()
        throughput_fig.show()

    @staticmethod
    def create_comparison_plot(
        results_list: list, labels: list, metric: str = "total"
    ) -> go.Figure:
        """
        Create comparison plot between multiple benchmark runs.

        Args:
            results_list: List of benchmark result dictionaries
            labels: Labels for each result set
            metric: Metric to compare ("total", "build", "query", etc.)

        Returns:
            Plotly figure comparing the runs
        """
        fig = go.Figure()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, (results, label) in enumerate(zip(results_list, labels)):
            config = results["config"]
            color = colors[i % len(colors)]

            # Add traces for each engine in this result set
            for engine_name, values in results[metric].items():
                fig.add_trace(
                    go.Scatter(
                        x=config.experiments,
                        y=values,
                        name=f"{label} - {engine_name}",
                        line={"color": color, "width": 3},
                        mode="lines+markers",
                    )
                )

        fig.update_layout(
            title=f"Benchmark Comparison - {metric.title()} Time",
            xaxis_title="Number of points",
            yaxis_title="Time (s)" if "rate" not in metric else "Ops/sec",
            template="plotly_dark",
            legend={"orientation": "v"},
            height=600,
        )

        if "rate" in metric:
            fig.update_yaxes(type="log")

        return fig
