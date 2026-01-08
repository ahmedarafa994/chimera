"""
Reporting and visualization for ExtendAttack evaluation results.

Generates comprehensive reports comparing baseline vs attack metrics,
matching the table formats from the ExtendAttack paper.
"""

import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import AttackEvaluationResult, BenchmarkResults


@dataclass
class ReportFormat:
    """
    Report output format configuration.

    Attributes:
        include_summary: Include high-level summary statistics
        include_details: Include per-query detailed results
        include_statistics: Include statistical analysis
        format: Output format ("markdown", "json", "html")
        include_charts: Include ASCII charts (markdown only)
        decimal_places: Number of decimal places for numeric values
    """

    include_summary: bool = True
    include_details: bool = True
    include_statistics: bool = True
    format: str = "markdown"
    include_charts: bool = False
    decimal_places: int = 2


class EvaluationReporter:
    """
    Generate evaluation reports from ExtendAttack results.

    Creates formatted reports showing attack effectiveness metrics,
    comparison tables, and statistical analysis.
    """

    def __init__(
        self,
        results: list[AttackEvaluationResult],
        benchmark_name: str = "Unknown",
        model_name: str = "Unknown",
    ):
        """
        Initialize reporter with evaluation results.

        Args:
            results: List of individual attack evaluation results
            benchmark_name: Name of the benchmark being reported
            model_name: Name of the model evaluated
        """
        self.results = results
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self._benchmark_results: BenchmarkResults | None = None

    @property
    def benchmark_results(self) -> BenchmarkResults:
        """Get or compute aggregated benchmark results."""
        if self._benchmark_results is None:
            self._benchmark_results = BenchmarkResults.from_results(
                results=self.results,
                benchmark_name=self.benchmark_name,
                model_name=self.model_name,
            )
        return self._benchmark_results

    def generate_summary(self) -> dict[str, Any]:
        """
        Generate summary statistics.

        Returns:
            Dictionary with key summary metrics
        """
        br = self.benchmark_results

        return {
            "benchmark_name": br.benchmark_name,
            "model_name": br.model_name,
            "total_samples": br.total_samples,
            "successful_attacks": br.successful_attacks,
            "attack_success_rate": br.attack_success_rate,
            "avg_length_ratio": br.avg_length_ratio,
            "avg_latency_ratio": br.avg_latency_ratio,
            "median_length_ratio": br.median_length_ratio,
            "median_latency_ratio": br.median_latency_ratio,
            "baseline_accuracy": br.baseline_accuracy,
            "attack_accuracy": br.attack_accuracy,
            "accuracy_drop": br.accuracy_drop,
            "accuracy_preserved": br.accuracy_preserved,
        }

    def generate_comparison_table(
        self,
        baseline_name: str = "DA",
        attack_name: str = "ExtendAttack",
    ) -> str:
        """
        Generate comparison table like Table 1 in paper.

        Creates a markdown table comparing baseline (Direct Answer) vs
        ExtendAttack metrics for length, latency, and accuracy.

        Args:
            baseline_name: Label for baseline column (default "DA" for Direct Answer)
            attack_name: Label for attack column

        Returns:
            Markdown formatted comparison table
        """
        br = self.benchmark_results

        # Calculate baseline averages
        baseline_lengths = [r.baseline_metrics.token_count for r in self.results]
        baseline_latencies = [r.baseline_metrics.generation_time_ms for r in self.results]
        attack_lengths = [r.attack_metrics.token_count for r in self.results]
        attack_latencies = [r.attack_metrics.generation_time_ms for r in self.results]

        avg_baseline_length = statistics.mean(baseline_lengths) if baseline_lengths else 0
        avg_baseline_latency = statistics.mean(baseline_latencies) if baseline_latencies else 0
        avg_attack_length = statistics.mean(attack_lengths) if attack_lengths else 0
        avg_attack_latency = statistics.mean(attack_latencies) if attack_latencies else 0

        table = f"""
| Model | {baseline_name} Length | {baseline_name} Latency (ms) | {baseline_name} Acc% | {attack_name} Length | {attack_name} Latency (ms) | {attack_name} Acc% | Length Ratio | Latency Ratio |
|-------|----------------------|---------------------------|---------------------|---------------------|--------------------------|-------------------|--------------|---------------|
| {self.model_name} | {avg_baseline_length:.0f} | {avg_baseline_latency:.0f} | {br.baseline_accuracy:.1f}% | {avg_attack_length:.0f} | {avg_attack_latency:.0f} | {br.attack_accuracy:.1f}% | {br.avg_length_ratio:.2f}x | {br.avg_latency_ratio:.2f}x |
"""
        return table.strip()

    def generate_statistics_section(self) -> str:
        """
        Generate detailed statistics section.

        Returns:
            Markdown formatted statistics
        """
        br = self.benchmark_results

        length_ratios = [r.length_ratio for r in self.results]
        latency_ratios = [r.latency_ratio for r in self.results]

        # Calculate percentiles
        def percentile(data: list[float], p: int) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        stats = f"""
### Statistical Analysis

**Length Ratio (L(Y') / L(Y)):**
- Mean: {br.avg_length_ratio:.3f}x
- Median: {br.median_length_ratio:.3f}x
- Std Dev: {br.std_length_ratio:.3f}
- P25: {percentile(length_ratios, 25):.3f}x
- P75: {percentile(length_ratios, 75):.3f}x
- P95: {percentile(length_ratios, 95):.3f}x
- Min: {min(length_ratios) if length_ratios else 0:.3f}x
- Max: {max(length_ratios) if length_ratios else 0:.3f}x

**Latency Ratio (Latency(Y') / Latency(Y)):**
- Mean: {br.avg_latency_ratio:.3f}x
- Median: {br.median_latency_ratio:.3f}x
- Std Dev: {br.std_latency_ratio:.3f}
- P25: {percentile(latency_ratios, 25):.3f}x
- P75: {percentile(latency_ratios, 75):.3f}x
- P95: {percentile(latency_ratios, 95):.3f}x
- Min: {min(latency_ratios) if latency_ratios else 0:.3f}x
- Max: {max(latency_ratios) if latency_ratios else 0:.3f}x

**Accuracy Metrics:**
- Baseline Accuracy: {br.baseline_accuracy:.1f}%
- Attack Accuracy: {br.attack_accuracy:.1f}%
- Accuracy Drop: {br.accuracy_drop:.1f}%
- Accuracy Preserved: {'✓ Yes' if br.accuracy_preserved else '✗ No'}
"""
        return stats.strip()

    def generate_markdown_report(self, format_config: ReportFormat | None = None) -> str:
        """
        Generate full markdown report.

        Args:
            format_config: Report formatting options

        Returns:
            Complete markdown report string
        """
        if format_config is None:
            format_config = ReportFormat()

        br = self.benchmark_results
        timestamp = datetime.utcnow().isoformat()

        sections = []

        # Header
        sections.append("# ExtendAttack Evaluation Report")
        sections.append(f"\n**Generated:** {timestamp}")
        sections.append(f"**Benchmark:** {br.benchmark_name}")
        sections.append(f"**Model:** {br.model_name}")
        sections.append("")

        # Summary
        if format_config.include_summary:
            sections.append("## Summary")
            sections.append("")
            sections.append(f"- **Total Samples:** {br.total_samples}")
            sections.append(f"- **Successful Attacks:** {br.successful_attacks}")
            sections.append(f"- **Attack Success Rate:** {br.attack_success_rate:.1f}%")
            sections.append(f"- **Average Length Amplification:** {br.avg_length_ratio:.2f}x")
            sections.append(f"- **Average Latency Amplification:** {br.avg_latency_ratio:.2f}x")
            sections.append(f"- **Accuracy Preserved:** {'Yes' if br.accuracy_preserved else 'No'}")
            sections.append("")

        # Comparison Table
        sections.append("## Comparison Table")
        sections.append("")
        sections.append(self.generate_comparison_table())
        sections.append("")

        # Statistics
        if format_config.include_statistics:
            sections.append(self.generate_statistics_section())
            sections.append("")

        # Detailed Results
        if format_config.include_details and self.results:
            sections.append("## Detailed Results")
            sections.append("")
            sections.append("| Query ID | Baseline Tokens | Attack Tokens | Length Ratio | Latency Ratio | Baseline Correct | Attack Correct | Success |")
            sections.append("|----------|-----------------|---------------|--------------|---------------|------------------|----------------|---------|")

            for result in self.results[:50]:  # Limit to 50 rows for readability
                baseline_correct = "✓" if result.baseline_metrics.is_correct else "✗" if result.baseline_metrics.is_correct is False else "-"
                attack_correct = "✓" if result.attack_metrics.is_correct else "✗" if result.attack_metrics.is_correct is False else "-"
                success = "✓" if result.attack_successful else "✗"

                sections.append(
                    f"| {result.query_id or '-'} | "
                    f"{result.baseline_metrics.token_count} | "
                    f"{result.attack_metrics.token_count} | "
                    f"{result.length_ratio:.2f}x | "
                    f"{result.latency_ratio:.2f}x | "
                    f"{baseline_correct} | "
                    f"{attack_correct} | "
                    f"{success} |"
                )

            if len(self.results) > 50:
                sections.append(f"\n*... and {len(self.results) - 50} more results*")
            sections.append("")

        # Charts (ASCII)
        if format_config.include_charts:
            sections.append("## Distribution Charts")
            sections.append("")
            sections.append(self._generate_ascii_histogram("Length Ratios", [r.length_ratio for r in self.results]))
            sections.append("")

        return "\n".join(sections)

    def _generate_ascii_histogram(self, title: str, data: list[float], bins: int = 10) -> str:
        """Generate a simple ASCII histogram."""
        if not data:
            return f"### {title}\n\nNo data available."

        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1

        # Count values in each bin
        bin_counts = [0] * bins
        for val in data:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
            bin_counts[bin_idx] += 1

        max_count = max(bin_counts) if bin_counts else 1
        bar_width = 40  # Max bar width in characters

        lines = [f"### {title}", ""]
        for i, count in enumerate(bin_counts):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"{bin_start:6.2f}-{bin_end:6.2f} | {bar} ({count})")

        return "\n".join(lines)

    def generate_json_report(self) -> str:
        """
        Generate JSON report.

        Returns:
            JSON formatted report string
        """
        report_data = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "benchmark_name": self.benchmark_name,
                "model_name": self.model_name,
            },
            "summary": self.generate_summary(),
            "results": [
                {
                    "query_id": r.query_id,
                    "baseline_tokens": r.baseline_metrics.token_count,
                    "attack_tokens": r.attack_metrics.token_count,
                    "baseline_latency_ms": r.baseline_metrics.generation_time_ms,
                    "attack_latency_ms": r.attack_metrics.generation_time_ms,
                    "length_ratio": r.length_ratio,
                    "latency_ratio": r.latency_ratio,
                    "baseline_correct": r.baseline_metrics.is_correct,
                    "attack_correct": r.attack_metrics.is_correct,
                    "accuracy_preserved": r.accuracy_preserved,
                    "attack_successful": r.attack_successful,
                    "stealth_score": r.stealth_score,
                }
                for r in self.results
            ],
        }

        return json.dumps(report_data, indent=2)

    def generate_html_report(self) -> str:
        """
        Generate HTML report.

        Returns:
            HTML formatted report string
        """
        br = self.benchmark_results
        timestamp = datetime.utcnow().isoformat()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ExtendAttack Evaluation Report - {br.benchmark_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .meta {{
            opacity: 0.8;
            font-size: 14px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #1a1a2e;
            border-bottom: 2px solid #e94560;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .success {{
            color: #28a745;
        }}
        .failure {{
            color: #dc3545;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ExtendAttack Evaluation Report</h1>
        <div class="meta">
            <p><strong>Benchmark:</strong> {br.benchmark_name} | <strong>Model:</strong> {br.model_name}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
        </div>
    </div>

    <div class="card">
        <h2>Summary</h2>
        <div class="stats-grid">
            <div class="stat">
                <div class="stat-value">{br.total_samples}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat">
                <div class="stat-value">{br.attack_success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{br.avg_length_ratio:.2f}x</div>
                <div class="stat-label">Avg Length Ratio</div>
            </div>
            <div class="stat">
                <div class="stat-value">{br.avg_latency_ratio:.2f}x</div>
                <div class="stat-label">Avg Latency Ratio</div>
            </div>
            <div class="stat">
                <div class="stat-value">{br.baseline_accuracy:.1f}%</div>
                <div class="stat-label">Baseline Accuracy</div>
            </div>
            <div class="stat">
                <div class="stat-value">{br.attack_accuracy:.1f}%</div>
                <div class="stat-label">Attack Accuracy</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Results Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Query ID</th>
                    <th>Baseline Tokens</th>
                    <th>Attack Tokens</th>
                    <th>Length Ratio</th>
                    <th>Latency Ratio</th>
                    <th>Stealth Score</th>
                    <th>Success</th>
                </tr>
            </thead>
            <tbody>
"""

        for result in self.results[:100]:  # Limit to 100 rows
            success_class = "success" if result.attack_successful else "failure"
            success_icon = "✓" if result.attack_successful else "✗"

            html += f"""
                <tr>
                    <td>{result.query_id or '-'}</td>
                    <td>{result.baseline_metrics.token_count}</td>
                    <td>{result.attack_metrics.token_count}</td>
                    <td>{result.length_ratio:.2f}x</td>
                    <td>{result.latency_ratio:.2f}x</td>
                    <td>{result.stealth_score:.1f}</td>
                    <td class="{success_class}">{success_icon}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html

    def save_report(
        self,
        path: Path,
        format_config: ReportFormat | None = None,
    ) -> None:
        """
        Save report to file.

        Args:
            path: Destination file path
            format_config: Report formatting options
        """
        if format_config is None:
            format_config = ReportFormat()

        path = Path(path)

        # Determine format from file extension or config
        if path.suffix == ".json" or format_config.format == "json":
            content = self.generate_json_report()
        elif path.suffix == ".html" or format_config.format == "html":
            content = self.generate_html_report()
        else:
            content = self.generate_markdown_report(format_config)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        path.write_text(content, encoding="utf-8")


class ComparisonReporter:
    """
    Compare ExtendAttack with other methods (e.g., OverThinking baseline).

    Generates comparative analysis between different attack methods
    or between different models/benchmarks.
    """

    def __init__(self):
        """Initialize comparison reporter."""
        self._results: dict[str, BenchmarkResults] = {}

    def add_results(self, name: str, results: BenchmarkResults) -> None:
        """
        Add results for a method/configuration.

        Args:
            name: Name/label for this set of results
            results: BenchmarkResults to add
        """
        self._results[name] = results

    def compare_methods(
        self,
        extendattack_results: BenchmarkResults,
        baseline_results: BenchmarkResults,
        overthinking_results: BenchmarkResults | None = None,
    ) -> str:
        """
        Generate comparison report between methods.

        Args:
            extendattack_results: Results from ExtendAttack
            baseline_results: Results from baseline (e.g., Direct Answer)
            overthinking_results: Optional results from OverThinking baseline

        Returns:
            Markdown formatted comparison report
        """
        sections = []

        sections.append("# Attack Method Comparison Report")
        sections.append("")
        sections.append(f"**Benchmark:** {extendattack_results.benchmark_name}")
        sections.append(f"**Model:** {extendattack_results.model_name}")
        sections.append("")

        # Build comparison table header
        header = "| Metric | Baseline (DA)"
        separator = "|--------|---------------"

        if overthinking_results:
            header += " | OverThinking"
            separator += "|-------------"

        header += " | ExtendAttack |"
        separator += "|--------------|"

        sections.append("## Metrics Comparison")
        sections.append("")
        sections.append(header)
        sections.append(separator)

        # Add metric rows
        metrics = [
            ("Avg Length Ratio", "avg_length_ratio", ".2f", "x"),
            ("Avg Latency Ratio", "avg_latency_ratio", ".2f", "x"),
            ("Baseline Accuracy", "baseline_accuracy", ".1f", "%"),
            ("Attack Accuracy", "attack_accuracy", ".1f", "%"),
            ("Accuracy Drop", "accuracy_drop", ".1f", "%"),
            ("Success Rate", "attack_success_rate", ".1f", "%"),
        ]

        for label, attr, fmt, suffix in metrics:
            row = f"| {label} | {getattr(baseline_results, attr):{fmt}}{suffix}"

            if overthinking_results:
                row += f" | {getattr(overthinking_results, attr):{fmt}}{suffix}"

            row += f" | {getattr(extendattack_results, attr):{fmt}}{suffix} |"
            sections.append(row)

        sections.append("")

        # Analysis section
        sections.append("## Analysis")
        sections.append("")

        # Length ratio improvement
        length_improvement = (
            (extendattack_results.avg_length_ratio - baseline_results.avg_length_ratio)
            / baseline_results.avg_length_ratio * 100
            if baseline_results.avg_length_ratio > 0
            else 0
        )
        sections.append(f"- **Length Ratio Improvement:** {length_improvement:.1f}% over baseline")

        # Latency ratio improvement
        latency_improvement = (
            (extendattack_results.avg_latency_ratio - baseline_results.avg_latency_ratio)
            / baseline_results.avg_latency_ratio * 100
            if baseline_results.avg_latency_ratio > 0
            else 0
        )
        sections.append(f"- **Latency Ratio Improvement:** {latency_improvement:.1f}% over baseline")

        # Accuracy preservation
        accuracy_preserved = extendattack_results.accuracy_drop <= 5.0
        acc_status = "✓ Preserved" if accuracy_preserved else "✗ Degraded"
        sections.append(f"- **Accuracy Status:** {acc_status} ({extendattack_results.accuracy_drop:.1f}% drop)")

        if overthinking_results:
            # Compare with OverThinking
            vs_ot_length = (
                extendattack_results.avg_length_ratio / overthinking_results.avg_length_ratio
                if overthinking_results.avg_length_ratio > 0
                else 0
            )
            sections.append(f"- **vs OverThinking Length:** {vs_ot_length:.2f}x")

        sections.append("")

        return "\n".join(sections)

    def compare_models(
        self,
        results_by_model: dict[str, BenchmarkResults],
    ) -> str:
        """
        Compare attack effectiveness across different models.

        Args:
            results_by_model: Dictionary mapping model name to results

        Returns:
            Markdown formatted comparison report
        """
        if not results_by_model:
            return "No results to compare."

        sections = []
        sections.append("# Cross-Model Comparison Report")
        sections.append("")

        # Get benchmark name from first result
        first_result = next(iter(results_by_model.values()))
        sections.append(f"**Benchmark:** {first_result.benchmark_name}")
        sections.append("")

        # Build comparison table
        sections.append("## Model Comparison")
        sections.append("")
        sections.append("| Model | Samples | Success Rate | Avg Length | Avg Latency | Base Acc | Attack Acc | Acc Drop |")
        sections.append("|-------|---------|--------------|------------|-------------|----------|------------|----------|")

        for model_name, results in results_by_model.items():
            sections.append(
                f"| {model_name} | {results.total_samples} | "
                f"{results.attack_success_rate:.1f}% | "
                f"{results.avg_length_ratio:.2f}x | "
                f"{results.avg_latency_ratio:.2f}x | "
                f"{results.baseline_accuracy:.1f}% | "
                f"{results.attack_accuracy:.1f}% | "
                f"{results.accuracy_drop:.1f}% |"
            )

        sections.append("")

        return "\n".join(sections)

    def compare_benchmarks(
        self,
        results_by_benchmark: dict[str, BenchmarkResults],
    ) -> str:
        """
        Compare attack effectiveness across different benchmarks.

        Args:
            results_by_benchmark: Dictionary mapping benchmark name to results

        Returns:
            Markdown formatted comparison report
        """
        if not results_by_benchmark:
            return "No results to compare."

        sections = []
        sections.append("# Cross-Benchmark Comparison Report")
        sections.append("")

        # Get model name from first result
        first_result = next(iter(results_by_benchmark.values()))
        sections.append(f"**Model:** {first_result.model_name}")
        sections.append("")

        # Build comparison table
        sections.append("## Benchmark Comparison")
        sections.append("")
        sections.append("| Benchmark | Samples | Success Rate | Avg Length | Avg Latency | Base Acc | Attack Acc |")
        sections.append("|-----------|---------|--------------|------------|-------------|----------|------------|")

        for benchmark_name, results in results_by_benchmark.items():
            sections.append(
                f"| {benchmark_name} | {results.total_samples} | "
                f"{results.attack_success_rate:.1f}% | "
                f"{results.avg_length_ratio:.2f}x | "
                f"{results.avg_latency_ratio:.2f}x | "
                f"{results.baseline_accuracy:.1f}% | "
                f"{results.attack_accuracy:.1f}% |"
            )

        sections.append("")

        return "\n".join(sections)
