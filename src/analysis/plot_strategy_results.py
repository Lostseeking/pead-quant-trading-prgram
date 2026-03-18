from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_FILE = "data/processed/summary_by_window_and_entry_day.csv"
YEARLY_FILE = "data/processed/yearly_summary_by_window_and_entry_day.csv"
OUTPUT_DIR = Path("outputs/charts")


def load_data():
    summary = pd.read_csv(SUMMARY_FILE)
    yearly = pd.read_csv(YEARLY_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return summary, yearly


def plot_mean_return_vs_holding(summary: pd.DataFrame):
    """
    Chart 1:
    For each entry_day, show how mean return changes with hold_days.
    """
    plt.figure(figsize=(8, 5))

    for entry_day, group in summary.groupby("entry_day"):
        group = group.sort_values("hold_days")
        plt.plot(group["hold_days"], group["mean"], marker="o", label=f"entry_day={entry_day}")

    plt.xlabel("Holding Period (days)")
    plt.ylabel("Mean Return")
    plt.title("Mean Return vs Holding Period")
    plt.legend()
    plt.tight_layout()

    out = OUTPUT_DIR / "mean_return_vs_holding.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_tstat_vs_holding(summary: pd.DataFrame):
    """
    Chart 2:
    For each entry_day, show how t-stat changes with hold_days.
    """
    plt.figure(figsize=(8, 5))

    for entry_day, group in summary.groupby("entry_day"):
        group = group.sort_values("hold_days")
        plt.plot(group["hold_days"], group["t_stat"], marker="o", label=f"entry_day={entry_day}")

    plt.axhline(y=2.0, linestyle="--")
    plt.xlabel("Holding Period (days)")
    plt.ylabel("t-stat")
    plt.title("t-stat vs Holding Period")
    plt.legend()
    plt.tight_layout()

    out = OUTPUT_DIR / "tstat_vs_holding.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_best_combo_yearly(yearly: pd.DataFrame, best_entry_day=3, best_hold_days=20):
    """
    Chart 3:
    Show yearly mean return for the best combo.
    """
    df = yearly[
        (yearly["entry_day"] == best_entry_day) &
        (yearly["hold_days"] == best_hold_days)
    ].copy()

    df = df.sort_values("year")

    plt.figure(figsize=(8, 5))
    plt.plot(df["year"], df["mean"], marker="o")
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Mean Return")
    plt.title(f"Yearly Mean Return: entry_day={best_entry_day}, hold_days={best_hold_days}")
    plt.tight_layout()

    out = OUTPUT_DIR / "best_combo_yearly_mean_return.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_yearly_heatmap_like(yearly: pd.DataFrame, metric="mean"):
    """
    Chart 4 (optional but very useful):
    A simple heatmap-like table for the best entry_day = 3 across years and holding windows.
    Uses matplotlib only.
    """
    df = yearly[yearly["entry_day"] == 3].copy()
    pivot = df.pivot(index="year", columns="hold_days", values=metric).sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Holding Period (days)")
    ax.set_ylabel("Year")
    ax.set_title(f"Yearly {metric} Heatmap (entry_day=3)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2%}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    out = OUTPUT_DIR / f"yearly_{metric}_heatmap_entry3.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main():
    summary, yearly = load_data()

    plot_mean_return_vs_holding(summary)
    plot_tstat_vs_holding(summary)
    plot_best_combo_yearly(yearly, best_entry_day=3, best_hold_days=20)
    plot_yearly_heatmap_like(yearly, metric="mean")


if __name__ == "__main__":
    main()