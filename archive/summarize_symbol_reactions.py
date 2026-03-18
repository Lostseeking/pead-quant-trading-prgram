import csv
from collections import defaultdict
from statistics import mean, stdev


def load_earnings_dataset(path="watchlist_earnings_research_3y.csv"):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            reaction_raw = row.get("reaction_1d", "").strip()
            if reaction_raw == "":
                continue

            try:
                reaction = float(reaction_raw)
            except ValueError:
                continue

            rows.append({
                "symbol": row["symbol"].strip().upper(),
                "date": row["date"].strip(),
                "year": int(row["year"]),
                "quarter": int(row["quarter"]),
                "reaction_1d": reaction,
            })

    return rows


def compute_symbol_stats(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["symbol"]].append(row["reaction_1d"])

    stats_rows = []

    for symbol, reactions in grouped.items():
        num_events = len(reactions)
        avg_reaction = mean(reactions)

        wins = sum(1 for r in reactions if r > 0)
        win_rate = wins / num_events if num_events > 0 else None

        volatility = stdev(reactions) if num_events >= 2 else 0.0

        stats_rows.append({
            "symbol": symbol,
            "num_events": num_events,
            "avg_reaction": avg_reaction,
            "win_rate": win_rate,
            "volatility": volatility,
        })

    stats_rows.sort(key=lambda x: x["avg_reaction"], reverse=True)
    return stats_rows


def save_stats_to_csv(stats_rows, output_path="earnings_reaction_summary.csv"):
    fieldnames = [
        "symbol",
        "num_events",
        "avg_reaction",
        "win_rate",
        "volatility",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in stats_rows:
            writer.writerow({
                "symbol": row["symbol"],
                "num_events": row["num_events"],
                "avg_reaction": round(row["avg_reaction"], 4) if row["avg_reaction"] is not None else "",
                "win_rate": round(row["win_rate"], 4) if row["win_rate"] is not None else "",
                "volatility": round(row["volatility"], 4) if row["volatility"] is not None else "",
            })


def print_top_bottom(stats_rows, top_n=10):
    print("\n=== Top Winners by Average Earnings Reaction ===")
    for row in stats_rows[:top_n]:
        print(
            f"{row['symbol']:>6} | "
            f"events={row['num_events']} | "
            f"avg={row['avg_reaction']:.2%} | "
            f"win_rate={row['win_rate']:.2%} | "
            f"vol={row['volatility']:.2%}"
        )

    print("\n=== Top Losers by Average Earnings Reaction ===")
    for row in stats_rows[-top_n:]:
        print(
            f"{row['symbol']:>6} | "
            f"events={row['num_events']} | "
            f"avg={row['avg_reaction']:.2%} | "
            f"win_rate={row['win_rate']:.2%} | "
            f"vol={row['volatility']:.2%}"
        )


def main():
    input_path = input(
        "Enter earnings dataset path (default watchlist_earnings_research_3y.csv): "
    ).strip()

    if not input_path:
        input_path = "watchlist_earnings_research_3y.csv"

    rows = load_earnings_dataset(input_path)
    print(f"\nLoaded {len(rows)} event rows.")

    stats_rows = compute_symbol_stats(rows)
    print(f"Computed stats for {len(stats_rows)} symbols.")

    print_top_bottom(stats_rows, top_n=10)

    output_path = "earnings_reaction_summary.csv"
    save_stats_to_csv(stats_rows, output_path)
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()