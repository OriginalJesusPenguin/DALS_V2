import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def format_table_df(df: pd.DataFrame) -> pd.DataFrame:
    # Map augmentation_type to no/yes
    df = df.copy()
    df['augmentation'] = df['augmentation_type'].replace({'noaug': 'no', 'aug': 'yes'})
    df['scaling'] = df['scaling_type']
    df['decoder'] = df['decoder_mode']
    df['latent_dim'] = df['latent_dim']
    df['ChamferL2 x 10000_mean'] = df['ChamferL2 x 10000_mean'].astype(float)
    # Round chamfer for readability
    df['chamfer'] = df['ChamferL2 x 10000_mean'].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")

    # Reorder/select columns
    table_df = df[['augmentation', 'scaling', 'decoder', 'latent_dim', 'chamfer']]

    # Sort for consistent display
    cat_aug = pd.CategoricalDtype(categories=['no', 'yes'], ordered=True)
    cat_scaling = pd.CategoricalDtype(categories=['individual', 'global'], ordered=True)
    cat_decoder = pd.CategoricalDtype(categories=['gcnn', 'mlp'], ordered=True)
    table_df['augmentation'] = table_df['augmentation'].astype(cat_aug)
    table_df['scaling'] = table_df['scaling'].astype(cat_scaling)
    table_df['decoder'] = table_df['decoder'].astype(cat_decoder)
    table_df = table_df.sort_values(['augmentation', 'scaling', 'decoder', 'latent_dim']).reset_index(drop=True)
    return table_df


def render_table_image(table_df: pd.DataFrame, title: str, out_path: str):
    sns.set_theme(style="whitegrid")

    # Create a figure sized to the number of rows
    n_rows = max(1, len(table_df))
    fig_height = min(0.35 * (n_rows + 2), 16)  # cap height for very long tables
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=12)

    # Build cell text
    col_labels = ['augmentation', 'scaling', 'decoder', 'latent_dim', 'ChamferL2 x 10000']
    cell_text = table_df.values.tolist()

    # Draw the table
    table = ax.table(cellText=cell_text,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(repo_root, 'model_inference_summary.csv')
    out_dir = os.path.join(repo_root, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(csv_path)

    # Validate required columns
    required = ['split_type', 'augmentation_type', 'scaling_type', 'decoder_mode', 'latent_dim', 'ChamferL2 x 10000_mean']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Filter and format for 'separate'
    df_sep_raw = df[df['split_type'] == 'separate'].copy()
    if len(df_sep_raw) > 0:
        df_sep = format_table_df(df_sep_raw)
        out_path_sep = os.path.join(out_dir, 'table_separate.png')
        render_table_image(df_sep, title="Results Table (split_type = 'separate')", out_path=out_path_sep)
        print(f"Saved: {out_path_sep}")
    else:
        print("No rows for split_type = 'separate'")

    # Filter and format for 'mixed'
    df_mix_raw = df[df['split_type'] == 'mixed'].copy()
    if len(df_mix_raw) > 0:
        df_mix = format_table_df(df_mix_raw)
        out_path_mix = os.path.join(out_dir, 'table_mixed.png')
        render_table_image(df_mix, title="Results Table (split_type = 'mixed')", out_path=out_path_mix)
        print(f"Saved: {out_path_mix}")
    else:
        print("No rows for split_type = 'mixed'")


if __name__ == '__main__':
    main()


