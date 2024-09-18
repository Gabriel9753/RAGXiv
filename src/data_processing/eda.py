import os
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_utils import load_data
from pyfonts import load_font

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import EDAConfig


def setup_font(cfg):
    if cfg.font_family == "Fira Sans":
        font = load_font("https://github.com/google/fonts/blob/main/ofl/firasans/FiraSans-Regular.ttf?raw=true")
        font = font.get_family()
    else:
        font = "sans-serif"
        cfg.font_family = "sans-serif"
    # TODO: Currently falling back to sans-serif every time and i dont know why
    print(f"Loaded font family: {font}")
    plt.rcParams["font.family"] = font
    plt.rcParams["font.size"] = cfg.font_size


def custom_countplot(df, column, cfg, top_n=None, title=None):
    # Set the style for the plot
    sns.set_theme(style=cfg.plot_style, palette=cfg.color_palette)

    plt.figure(figsize=cfg.figure_size)

    # Prepare data
    if top_n:
        value_counts = df[column].value_counts().nlargest(top_n)
        data = df[df[column].isin(value_counts.index)]
    else:
        data = df

    # Create count plot
    ax = sns.countplot(x=column, data=data, color=cfg.primary_color, order=value_counts.index if top_n else None)

    # Customize the plot
    if title is None:
        plt.title(f"Distribution of {column}", fontsize=cfg.title_font_size)
    else:
        plt.title(title, fontsize=cfg.title_font_size)
    plt.xlabel(column, fontsize=cfg.label_font_size)
    plt.ylabel("Count", fontsize=cfg.label_font_size)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=cfg.x_tick_rotation)

    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Save the plot
    plt.tight_layout()
    filename = f'{column}_distribution{"_top_"+str(top_n) if top_n else ""}.png'
    plt.savefig(os.path.join(cfg.eda_dir, filename), dpi=cfg.dpi, bbox_inches="tight")
    plt.close()


def main():
    cfg = EDAConfig()
    setup_font(cfg)
    paper_df, paper_authors, paper_references = load_data(drop_missing=cfg.drop_missing)
    print(f"Loaded {len(paper_df)} papers!")
    print(f"Columns in df: {paper_df.columns.values}")

    os.makedirs(cfg.eda_dir, exist_ok=True)

    # count plots for different columns
    custom_countplot(paper_df, "author_count", cfg, title="Distribution of authors per paper", top_n=15)
    custom_countplot(paper_df, "reference_count", cfg, top_n=15)
    custom_countplot(paper_df, "citation_count", cfg, top_n=15)
    custom_countplot(paper_df, "update_year", cfg, top_n=15)
    custom_countplot(paper_df, "title_words", cfg, top_n=15)
    custom_countplot(paper_df, "abstract_words", cfg, top_n=15)


if __name__ == "__main__":
    main()
