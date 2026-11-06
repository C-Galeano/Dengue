"""End-to-end run: case records -> weekly series -> features -> model comparison.

Usage:
    python -m dengue.pipeline data/raw/bucaramanga.csv --test-weeks 26
    python -m dengue.pipeline data/raw/bucaramanga.csv --climate data/external/ideam_daily.csv
"""
import argparse

import pandas as pd

from dengue.config import FIGURES_DIR, PROCESSED_DIR
from dengue.data import load_cases
from dengue.evaluation import evaluate, plot_real_vs_predicted, time_split
from dengue.features import build_features, build_weekly_series, climate_to_epiweeks
from dengue.models import default_models


def run(cases_path, climate_path=None, test_weeks=26):
    weekly = build_weekly_series(load_cases(cases_path))
    climate = climate_to_epiweeks(pd.read_csv(climate_path)) if climate_path else None
    features = build_features(weekly, climate)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DIR / "weekly_features.csv")

    train, test = time_split(features, test_weeks)
    metrics, predictions = evaluate(default_models(), train, test)
    plot_real_vs_predicted(predictions, train, FIGURES_DIR / "real_vs_predicted.png")
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", help="SIVIGILA CSV (one row per case)")
    parser.add_argument("--climate", help="Daily IDEAM CSV with a 'date' column and numeric variables")
    parser.add_argument("--test-weeks", type=int, default=26)
    args = parser.parse_args()

    metrics, _ = run(args.cases, args.climate, args.test_weeks)
    print(metrics.round(2).to_string())


if __name__ == "__main__":
    main()
