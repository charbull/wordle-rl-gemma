
from pathlib import Path
import argparse
from src.utils.logging import plot_cumulative_wins, plot_comparison_chart, plot_cumulative_wins_sxs, plot_win_distribution_sxs
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a cumulative wins plot for training and evaluation from a .jsonl metrics file."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the training_metrics.jsonl file to be plotted."
    )
    parser.add_argument(
        "--eval",
        type=bool,
        required=False,
        default=False,
        help="If set to true runs side by side plots"
    )
    args = parser.parse_args()
    metrics_file_path = Path(args.file)
    is_eval = args.eval

    if not is_eval:
        plot_cumulative_wins(metrics_file_path)
        plot_comparison_chart(metrics_file_path)  # Optional: If you want to plot comparison chart as well
    else:
        plot_cumulative_wins_sxs(metrics_file_path)
        plot_win_distribution_sxs(metrics_file_path)
        plt.show()