"""
Generate sample data for testing the FinBERT-CNN pipeline
Creates synthetic headlines and RV values

Usage:
    python generate_sample_data.py --n_samples 1000 --output data/headlines_rv.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Sample financial headline templates
HEADLINE_TEMPLATES = [
    "{company} reports {adj} quarterly earnings, {verb} analyst expectations",
    "{company} stock {movement} {pct}% on {news}",
    "Federal Reserve {action}, {impact} expected",
    "{company} announces {event}, shares {reaction}",
    "{sector} stocks {movement} amid {concern}",
    "Market volatility {increases} as {reason}",
    "{company} CEO {ceo_action}, stock price {reaction}",
    "Trading volume {volume_adj} for {company} on {news}",
    "{economic_indicator} {direction}, markets {reaction}",
    "{company} faces {challenge}, analysts {analyst_action}",
]

# Vocabulary
COMPANIES = ["Apple", "Microsoft", "Amazon", "Tesla", "Google", "Meta", "Nvidia"]
SECTORS = ["Tech", "Financial", "Energy", "Healthcare", "Consumer"]
ADJECTIVES = ["strong", "weak", "record", "disappointing", "surprising"]
VERBS = ["beating", "missing", "meeting", "exceeding"]
MOVEMENTS = ["surges", "plunges", "climbs", "falls", "rallies", "drops"]
REACTIONS = ["soar", "tumble", "rise", "fall", "jump", "decline"]
CONCERNS = [
    "recession fears",
    "inflation concerns",
    "supply chain issues",
    "geopolitical tensions",
]
ACTIONS = ["raises rates", "cuts rates", "maintains policy", "signals dovish stance"]
IMPACTS = ["volatility", "stability", "uncertainty", "growth"]
NEWS = ["earnings report", "merger news", "regulatory approval", "product launch"]
EVENTS = ["stock split", "buyback program", "dividend increase", "restructuring"]
CEO_ACTIONS = ["resigns", "appointed", "announces strategy shift"]
ECONOMIC_INDICATORS = ["GDP", "Inflation", "Employment", "Retail sales"]
DIRECTIONS = ["rises", "falls", "exceeds expectations", "disappoints"]
CHALLENGES = ["regulatory scrutiny", "competition", "lawsuit", "scandal"]
ANALYST_ACTIONS = ["downgrade", "upgrade", "maintain rating", "revise outlook"]


def generate_headline():
    """Generate a random financial headline"""
    template = np.random.choice(HEADLINE_TEMPLATES)

    headline = template.format(
        company=np.random.choice(COMPANIES),
        adj=np.random.choice(ADJECTIVES),
        verb=np.random.choice(VERBS),
        movement=np.random.choice(MOVEMENTS),
        pct=np.random.randint(1, 15),
        news=np.random.choice(NEWS),
        action=np.random.choice(ACTIONS),
        impact=np.random.choice(IMPACTS),
        event=np.random.choice(EVENTS),
        reaction=np.random.choice(REACTIONS),
        sector=np.random.choice(SECTORS),
        concern=np.random.choice(CONCERNS),
        ceo_action=np.random.choice(CEO_ACTIONS),
        volume_adj=np.random.choice(["spikes", "drops", "doubles"]),
        economic_indicator=np.random.choice(ECONOMIC_INDICATORS),
        direction=np.random.choice(DIRECTIONS),
        challenge=np.random.choice(CHALLENGES),
        analyst_action=np.random.choice(ANALYST_ACTIONS),
        increases=np.random.choice(["increases", "rises", "spikes", "surges"]),
        reason=np.random.choice(CONCERNS),
    )

    return headline


def generate_rv(base_rv=2.0, volatility=1.0, headline=None):
    """
    Generate realized volatility with some correlation to headline sentiment

    Args:
        base_rv: Base RV value
        volatility: Standard deviation
        headline: Headline text (optional, for sentiment-based adjustment)

    Returns:
        rv: Realized volatility value
    """
    # Base RV with random noise
    rv = np.random.lognormal(mean=np.log(base_rv), sigma=volatility)

    # Add sentiment-based adjustment (simple keyword matching)
    if headline:
        # Words that increase volatility
        high_vol_words = [
            "plunge",
            "surge",
            "crisis",
            "crash",
            "soar",
            "tumble",
            "spike",
            "record",
            "disappointing",
            "recession",
        ]
        # Words that decrease volatility
        low_vol_words = ["stable", "steady", "maintain", "unchanged", "consistent"]

        headline_lower = headline.lower()

        # Count keyword occurrences
        high_vol_count = sum(1 for word in high_vol_words if word in headline_lower)
        low_vol_count = sum(1 for word in low_vol_words if word in headline_lower)

        # Adjust RV based on sentiment
        sentiment_factor = 1.0 + 0.3 * (high_vol_count - low_vol_count)
        rv *= sentiment_factor

    # Ensure RV is positive
    rv = max(0.1, rv)

    return rv


def generate_dataset(
    n_samples=1000, start_date="2020-01-01", base_rv=2.0, volatility=1.0
):
    """
    Generate synthetic dataset

    Args:
        n_samples: Number of samples
        start_date: Start date for time series
        base_rv: Base realized volatility
        volatility: Volatility of RV process

    Returns:
        df: DataFrame with columns [date, headline, realized_volatility]
    """
    print(f"Generating {n_samples} samples...")

    # Generate dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range(n_samples)]

    # Generate headlines and RV
    data = []
    for i, date in enumerate(dates):
        headline = generate_headline()

        # Add some temporal dynamics (trending volatility)
        trend = 1.0 + 0.3 * np.sin(2 * np.pi * i / 252)  # Annual cycle
        current_base_rv = base_rv * trend

        rv = generate_rv(current_base_rv, volatility, headline)

        data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "headline": headline,
                "realized_volatility": round(rv, 4),
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples} samples...")

    df = pd.DataFrame(data)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Samples: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  RV stats:")
    print(f"    Mean: {df['realized_volatility'].mean():.4f}")
    print(f"    Std:  {df['realized_volatility'].std():.4f}")
    print(f"    Min:  {df['realized_volatility'].min():.4f}")
    print(f"    Max:  {df['realized_volatility'].max():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for testing")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--output", type=str, default="data/headlines_rv.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--start_date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--base_rv", type=float, default=2.0, help="Base realized volatility"
    )
    parser.add_argument(
        "--volatility", type=float, default=1.0, help="Volatility of RV process"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Generate dataset
    df = generate_dataset(
        n_samples=args.n_samples,
        start_date=args.start_date,
        base_rv=args.base_rv,
        volatility=args.volatility,
    )

    # Save to CSV
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # Show sample
    print("\nSample data:")
    print(df.head(10))


if __name__ == "__main__":
    main()
