import glob
import os
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List

import pandas as pd


IMPORTANCE_GLOB = '/home/ralbe/DALS/mesh_autodecoder/relax_explanations/importance_*_*_testing.csv'
SUMMARY_DISPLAY_LIMIT = None  # set to an integer to limit the aggregated table output


def infer_cohort(filename: str) -> str:
    """Infer cohort label from a filename like importance_{cohort}_{id}_testing.csv."""
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 3:
        return parts[1]
    return 'unknown'


def build_summary(
    rank_map: Dict[int, List[int]],
    importance_map: Dict[int, List[float]],
) -> pd.DataFrame:
    rows = []
    for latent_index in sorted(rank_map.keys()):
        ranks = rank_map[latent_index]
        if not ranks:
            continue
        importances = importance_map.get(latent_index, [])
        if not importances:
            continue
        rows.append(
            {
                'latent_index': latent_index,
                'count': len(ranks),
                'mean_rank': sum(ranks) / len(ranks),
                'best_rank': min(ranks),
                'worst_rank': max(ranks),
                'mean_importance': sum(importances) / len(importances),
                'max_importance': max(importances),
                'min_importance': min(importances),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(by=['mean_rank', 'mean_importance'], ascending=[True, False])
        .reset_index(drop=True)
    )


def display_summary_table(summary: pd.DataFrame, header: str) -> None:
    if summary.empty:
        print(f'\n{header}: no data available.')
        return

    display_df = summary
    if SUMMARY_DISPLAY_LIMIT is not None:
        display_df = summary.head(SUMMARY_DISPLAY_LIMIT)

    print(f'\n{header}:')
    print(display_df.to_string(index=False, float_format='{:.6f}'.format))

    if SUMMARY_DISPLAY_LIMIT is not None and SUMMARY_DISPLAY_LIMIT < len(summary):
        print(
            f'\nShowing top {SUMMARY_DISPLAY_LIMIT} latent indices out of {len(summary)}.'
        )


def main() -> None:
    importance_files = sorted(glob.glob(IMPORTANCE_GLOB))

    if not importance_files:
        print(f'No files found matching pattern: {IMPORTANCE_GLOB}')
        return

    aggregate_ranks: DefaultDict[int, List[int]] = defaultdict(list)
    aggregate_importances: DefaultDict[int, List[float]] = defaultdict(list)
    cohort_ranks: DefaultDict[str, DefaultDict[int, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cohort_importances: DefaultDict[str, DefaultDict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for importance_file in importance_files:
        df = pd.read_csv(importance_file)

        if 'importance' not in df.columns or 'latent_index' not in df.columns:
            print(f'Skipping {importance_file}: required columns missing.')
            continue

        ranked = (
            df[['latent_index', 'importance']]
            .sort_values(by='importance', ascending=False)
            .reset_index(drop=True)
        )
        ranked['rank'] = ranked.index + 1

        cohort = infer_cohort(importance_file)

        for _, row in ranked.iterrows():
            latent_idx = int(row['latent_index'])
            rank_value = int(row['rank'])
            importance_value = float(row['importance'])

            aggregate_ranks[latent_idx].append(rank_value)
            aggregate_importances[latent_idx].append(importance_value)
            cohort_ranks[cohort][latent_idx].append(rank_value)
            cohort_importances[cohort][latent_idx].append(importance_value)

        print(f'\nRanking of latent indices for {os.path.basename(importance_file)}:')
        print(ranked[['rank', 'latent_index', 'importance']].to_string(index=False))

    if not aggregate_ranks:
        return

    overall_summary = build_summary(aggregate_ranks, aggregate_importances)
    display_summary_table(
        overall_summary, 'Aggregated latent index statistics (all cohorts)'
    )

    for cohort in sorted(cohort_ranks.keys()):
        cohort_summary = build_summary(cohort_ranks[cohort], cohort_importances[cohort])
        header = f'Aggregated latent index statistics ({cohort} cohort)'
        display_summary_table(cohort_summary, header)


if __name__ == '__main__':
    main()