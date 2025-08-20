import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

def plot_cumulative_wins(metrics_file: Path):
    """
    Reads a .jsonl metrics file and plots the cumulative number of wins
    for both training and evaluation games over training steps.

    Args:
        metrics_file: Path to the training_metrics.jsonl file.
    """
    if not metrics_file.exists():
        print(f"Error: Metrics file not found at '{metrics_file}'")
        return

    print(f"Loading metrics from: {metrics_file}")
    with open(metrics_file, 'r') as f:
        records = [json.loads(line) for line in f]

    if not records:
        print("Metrics file is empty. No plot generated.")
        return

    df = pd.DataFrame(records)

    # --- 1. Process Training Data ---
    train_df = df[df['log_type'] == 'train'].copy()
    if not train_df.empty:
        train_df = train_df.sort_values(by='step')
        train_df['is_win'] = train_df['solved'].astype(int)
        train_df['cumulative_wins'] = train_df['is_win'].cumsum()
    
    # --- 2. Process Evaluation Data ---
    eval_df = df[df['log_type'] == 'eval'].copy()
    if not eval_df.empty:
        # For evals, a 'step' can have multiple games. We need to group by step.
        eval_wins_by_step = eval_df.groupby('step')['solved'].sum().astype(int)
        eval_df = eval_wins_by_step.cumsum().reset_index()
        eval_df.rename(columns={'solved': 'cumulative_wins'}, inplace=True)

    # --- 3. Generate the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot Training Wins
    if not train_df.empty:
        ax.plot(train_df['step'], train_df['cumulative_wins'], 
                label='Cumulative Training Wins', color='dodgerblue', linewidth=2)
        # Add annotation for final training wins
        final_train_step = train_df['step'].iloc[-1]
        final_train_wins = train_df['cumulative_wins'].iloc[-1]
        ax.text(final_train_step, final_train_wins, f' {final_train_wins} Train Wins', 
                verticalalignment='bottom', color='dodgerblue')

    # Plot Evaluation Wins
    if not eval_df.empty:
        ax.plot(eval_df['step'], eval_df['cumulative_wins'], 
                label='Cumulative Evaluation Wins', color='orangered', 
                linestyle='--', marker='o', markersize=5)
        # Add annotation for final evaluation wins
        final_eval_step = eval_df['step'].iloc[-1]
        final_eval_wins = eval_df['cumulative_wins'].iloc[-1]
        ax.text(final_eval_step, final_eval_wins, f' {final_eval_wins} Eval Wins', 
                verticalalignment='top', color='orangered')

    # Formatting the plot
    ax.set_title('Cumulative Wins During Training vs. Evaluation', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Total Number of Wins', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()

    # --- 4. Save the plot ---
    input_filename_stem = metrics_file.stem 
    plot_filename = metrics_file.parent / f"cumulative_wins_train_vs_eval_{input_filename_stem}.png"
    plt.savefig(plot_filename)
    
    print(f"Successfully generated and saved plot to '{plot_filename}'")
    plt.show()


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