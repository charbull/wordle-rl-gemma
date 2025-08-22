
from pathlib import Path
import argparse
from utils.logging import plot_cumulative_wins



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
    args = parser.parse_args()
    metrics_file_path = Path(args.file)
    plot_cumulative_wins(metrics_file_path)