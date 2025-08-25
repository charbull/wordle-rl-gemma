from typing import List
import matplotlib.pyplot as plt
from src.wordle.game import GameRecord, GameRollout
from pathlib import Path
import json
from dataclasses import asdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def plot_training_curves(
    timestamp: str,
    train_steps: List[int],
    train_losses: List[float],
    train_avg_rewards: List[float],
    eval_steps: List[int],
    eval_win_rates: List[float],
    dir_path: str
):
    """Generates and saves a plot with training and evaluation curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Training Loss
    ax1.plot(train_steps, train_losses, label="Training Loss", color='tab:blue')
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title(f"Training and Evaluation Curves ({timestamp})")

    # Panel 2: Rewards and Win Rate
    ax2.plot(train_steps, train_avg_rewards, label="Avg. Training Reward", color='tab:green')
    ax2.plot(eval_steps, eval_win_rates, label="Eval Win Rate (%)", color='tab:red', marker='o', linestyle='--')
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Reward / Win Rate (%)")
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plot_filename = f"{dir_path}/training_curves_{timestamp}.png"
    plt.savefig(plot_filename)
    print(f"Training curve plot saved to {plot_filename}")


def plot_comparison_chart(jsonl_path: str):
    """Generates and saves a bar chart comparing model win rates.
    
    Args:
        jsonl_path: Path to the .jsonl file containing game results.
    
    """
    # --- Process and Print Final Results ---
    all_results = read_metrics_file(jsonl_path)
    results_df = pd.DataFrame(all_results)
    
    # Use 'log_type' to group data by model
    summary = results_df.groupby('log_type').agg(
        total_wins=('solved', lambda x: x.sum()),
        total_games=('solved', 'count'),
        avg_turns_on_win=('turns_to_solve', lambda x: x[results_df.loc[x.index, 'solved']].mean())
    ).reset_index()
    summary['win_rate'] = (summary['total_wins'] / summary['total_games']) * 100
    
    print("\n" + "="*60 + "\n" + " " * 18 + "SIDE-BY-SIDE EVALUATION RESULTS" + "\n" + "="*60)
    for _, row in summary.iterrows():
        print(f"\n--- {row['log_type']} ---") # Use log_type for model name
        print(f"  Win Rate: {row['win_rate']:.2f}% ({row['total_wins']}/{row['total_games']})")
        print(f"  Avg. Turns on Win: {row['avg_turns_on_win']:.2f}")
    print("\n" + "="*60)
    # Since we know both models played the same number of games, we can
    # simply count the number of unique secret words to find this value.
    if not results_df.empty:
        num_games = results_df['secret_word'].nunique()
    else:
        num_games = 0
    
    plot_title = (
        'Model Performance Comparison: Wordle Win Rate\n'
        f'(Based on {num_games} games per model)'
    )

    # Calculate summary stats
    summary = results_df.groupby('log_type')['solved'].value_counts(normalize=True).unstack(fill_value=0)
    summary['win_rate'] = summary.get(True, 0) * 100
    
    model_names = summary.index
    win_rates = summary['win_rate']
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bars = ax.bar(model_names, win_rates, color=['skyblue', 'orangered'])
    
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path(jsonl_path).parent
    plot_filename = output_dir / f"model_comparison_wins_num_games_{num_games}.png"
    plt.savefig(plot_filename)
    print(f"\n📈 Comparison plot saved to '{plot_filename}'")
    plt.show()

def read_metrics_file(jsonl_path):
    with open(jsonl_path, 'r') as f:
        all_results = [json.loads(line) for line in f]
    return all_results

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
    plot_filename = metrics_file.parent / f"plots/cumulative_wins_train_vs_eval_{input_filename_stem}.png"
    plt.savefig(plot_filename)
    
    print(f"Successfully generated and saved plot to '{plot_filename}'")
    plt.show()


def log_game_result(step: int, loss: float, game_rollout: GameRollout, log_type: str) -> GameRecord:
    """
    Analyzes a completed solved game and returns a structured GameRecord object.
    Args:
        step: Current training step.
        loss: Loss value at the current step.
        game_rollout: The completed GameRollout object containing game details.
        log_type: A string indicating the type of log (e.g., "train", "eval").
    Returns:
        A GameRecord dataclass instance summarizing the game outcome.
    """
    if game_rollout.solved:
        winning_attempt = game_rollout.attempts[-1]
        final_score = winning_attempt.training_reward
        num_turns = len(set(att.prompt_string for att in game_rollout.attempts))
    else:
        final_score = -1.0 
        num_turns = -1

    return GameRecord(
        log_type=log_type,
        step=step,
        solved=game_rollout.solved,
        secret_word=game_rollout.secret_word,
        turns_to_solve=num_turns,
        final_reward=final_score,
        loss_at_step=loss
    )


def write_metrics_to_file(records: List[GameRecord], output_file: Path):
    """
    Appends a list of record dictionaries to a .jsonl file.
    Creates the necessary parent directory if it does not exist.

    Args:
        records: A list of dictionaries, where each dict is a game result.
        output_file: The Path object for the output file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'a') as f:
        for record in records:
            json_record = json.dumps(asdict(record))
            f.write(json_record + '\n')

def log_metrics_to_tensorboard(writer: SummaryWriter, records: List[GameRecord], global_step: int, log_type: str):
    """
    Calculates summary statistics from a list of GameRecords and logs them to TensorBoard.
    """
    if not records:
        return

    total_games = len(records)
    wins = sum(1 for r in records if r.solved) # Changed to r.solved
    win_rate = (wins / total_games) * 100.0 if total_games > 0 else 0.0
    
    winning_turns = [r.turns_to_solve for r in records if r.solved] # Changed to r.turns_to_solve
    avg_turns_on_win = np.mean(winning_turns) if winning_turns else 0.0
    avg_final_reward = np.mean([r.final_reward for r in records if r.solved]) # Changed to r.final_reward

    writer.add_scalar(f"Performance/{log_type}_win_rate_percent", win_rate, global_step)
    if winning_turns:
        writer.add_scalar(f"Performance/{log_type}_avg_turns_on_win", avg_turns_on_win, global_step)
        writer.add_scalar(f"Performance/{log_type}_avg_win_reward", avg_final_reward, global_step)
        writer.add_histogram(f"Distributions/{log_type}_turns_to_solve", np.array(winning_turns), global_step)

    text_summary = f"### {log_type.capitalize()} Summary @ Step {global_step}\n| Metric | Value |\n|---|---|\n"
    text_summary += f"| Win Rate | {win_rate:.2f}% ({wins}/{total_games}) |\n| Avg. Turns on Win | {avg_turns_on_win:.2f} |\n"
    text_summary += "\n**Recent Wins:**\n| Secret Word | Turns |\n|---|---|\n"
    recent_wins = [r for r in records if r.solved][-5:]
    for win in recent_wins:
        text_summary += f"| {win.secret_word} | {win.turns_to_solve} |\n"
    writer.add_text(f"Summaries/{log_type}_game_summary", text_summary, global_step)

def truncate_jsonl_log(log_file: Path, max_step: int):
    """
    Reads a .jsonl log file, keeps all records up to a maximum step,
    and overwrites the file with the truncated data. Creates a backup first.
    """
    if not log_file.exists():
        print(f"Log file '{log_file}' not found. Nothing to truncate.")
        return

    # Create a backup file path
    backup_file = log_file.with_suffix(log_file.suffix + '.bak')
    
    # If a backup already exists, we assume the log is already truncated
    if backup_file.exists():
        print(f"Backup file '{backup_file}' already exists. Assuming log is clean.")
        return
        
    log_file.rename(backup_file)
    print(f"Backed up original log file to: {backup_file}")

    records_kept_count = 0
    with open(backup_file, 'r') as f_in, open(log_file, 'w') as f_out:
        for line in f_in:
            try:
                record = json.loads(line)
                # Keep the record if its step is at or before the resume step
                if record.get('step', 0) < max_step:
                    f_out.write(line)
                    records_kept_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in log: {line.strip()}")

    print(f"Successfully truncated log file. Kept {records_kept_count} records up to step {max_step}.")



def plot_cumulative_wins_sxs(metrics_file_path: str):
    """
    Generates a step chart showing cumulative wins over game steps.
    """
    if not metrics_file_path:
        print("No logs to process.")
        return

    log_list = read_metrics_file(metrics_file_path)

    # Sort logs by step to ensure correct order
    log_list.sort(key=lambda x: x['step'])
    steps = sorted(list(set(log['step'] for log in log_list)))
    
    base_wins_over_time = []
    lora_wins_over_time = []
    
    base_win_count = 0
    lora_win_count = 0
    
    for step in steps:
        for log in log_list:
            if log['step'] == step:
                if log['solved']:
                    if log['log_type'] == 'Base Model':
                        base_win_count += 1
                    elif log['log_type'] == 'LoRA Model':
                        lora_win_count += 1
        base_wins_over_time.append(base_win_count)
        lora_wins_over_time.append(lora_win_count)

    plt.figure(figsize=(12, 7))
    plt.step(steps, base_wins_over_time, where='post', label='Base Model Wins', color='cornflowerblue', linewidth=2)
    plt.step(steps, lora_wins_over_time, where='post', label='LoRA Model Wins', color='salmon', linewidth=2)
    
    title_text = f'Cumulative Wordle Wins (Total Games Played: {len(steps)})'
    plt.title(title_text, fontsize=16)

    plt.xlabel('Game Step', fontsize=12)
    plt.ylabel('Total Cumulative Wins', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    max_wins = max(base_win_count, lora_win_count)
    if max_wins > 0:
        plt.yticks(range(0, max_wins + 5, 5))

    output_path = Path(metrics_file_path).parent / f"cumulative_wins_sxs_lora_base_{len(steps)}_games.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {output_path}")
    plt.show()


def plot_win_distribution_sxs(metrics_file_path: str):
    """
    Generates a bar chart showing the distribution of wins by number of turns.
    """

    log_list = read_metrics_file(metrics_file_path)

    base_turns = [log['turns_to_solve'] for log in log_list if log['log_type'] == 'Base Model' and log['solved']]
    lora_turns = [log['turns_to_solve'] for log in log_list if log['log_type'] == 'LoRA Model' and log['solved']]

    if not base_turns and not lora_turns:
        print("No wins recorded, skipping distribution chart.")
        return

    base_counts = Counter(base_turns)
    lora_counts = Counter(lora_turns)
    
    labels = sorted(list(set(base_counts.keys()) | set(lora_counts.keys())))
    
    base_values = [base_counts.get(turn, 0) for turn in labels]
    lora_values = [lora_counts.get(turn, 0) for turn in labels]
    steps = sorted(list(set(log['step'] for log in log_list)))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, base_values, width, label='Base Model', color='cornflowerblue')
    rects2 = ax.bar(x + width/2, lora_values, width, label='LoRA Model', color='salmon')

    ax.set_ylabel('Number of Wins', fontsize=12)
    ax.set_xlabel('Turns to Solve', fontsize=12)
    ax.set_title(f'Distribution of Wins by Number of Turns (Total Games Played {len(steps)})', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()

    output_path = Path(metrics_file_path).parent / f"win_distribution_sxs_lora_base_{len(steps)}_games.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {output_path}")
    plt.show()