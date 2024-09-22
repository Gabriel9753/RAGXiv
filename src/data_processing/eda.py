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
    """
    Set up the font for matplotlib plots based on the configuration.

    This function attempts to load a custom font (Fira Sans) or falls back to sans-serif.
    It also sets global matplotlib parameters for font family and size.

    Args:
        cfg (EDAConfig): Configuration object containing font settings.
    """
    if cfg.font_family == "Fira Sans":
        font = load_font("https://github.com/google/fonts/blob/main/ofl/firasans/FiraSans-Regular.ttf?raw=true")
        font = font.get_family()
    else:
        font = "sans-serif"
        cfg.font_family = "sans-serif"
    print(f"Loaded font family: {font}")
    plt.rcParams["font.family"] = font
    plt.rcParams["font.size"] = cfg.font_size


def setup_dark_theme_transparent():
    """
    Set up the dark theme with a transparent background for Seaborn and Matplotlib.
    """
    sns.set_theme(style="darkgrid", palette="deep")

    # Matplotlib customizations for a dark theme with transparency
    plt.rcParams.update({
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.facecolor": "none",  # Transparent plot background
        "axes.edgecolor": "white",
        "figure.facecolor": "none",  # Transparent figure background
        "grid.color": "#2a2a2a",  # Darker grid lines
        "axes.grid": True,
        "font.size": 12,
        "legend.facecolor": "none",  # Transparent legend background
        "savefig.transparent": True,  # Transparent background when saving
    })

def custom_countplot(df, column, cfg, top_n=None, title=None):
    """
    Create and save a custom count plot for a specified column in the DataFrame.

    This function creates a count plot using seaborn, customizes it based on the provided
    configuration, and saves it as an image file.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        column (str): The name of the column to plot.
        cfg (EDAConfig): Configuration object containing plot settings.
        top_n (int, optional): If provided, plot only the top n categories.
        title (str, optional): Custom title for the plot. If None, a default title is used.
    """
    # Set the style for the plot
    # sns.set_theme(style=cfg.plot_style, palette=cfg.color_palette)
    setup_dark_theme_transparent()

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
        plt.title(f"Distribution of {column}", fontsize=cfg.title_font_size, color="white")
    else:
        plt.title(title, fontsize=cfg.title_font_size, color="white")
    plt.xlabel(column, fontsize=cfg.label_font_size, color="white")
    plt.ylabel("Count", fontsize=cfg.label_font_size, color="white")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=cfg.x_tick_rotation, color="white")
    plt.yticks(color="white")

    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            color="white"  # Text color for annotation
        )

    # Save the plot with transparency
    plt.tight_layout()
    filename = f'{column}_distribution{"_top_"+str(top_n) if top_n else ""}.png'
    plt.savefig(os.path.join(cfg.eda_dir, filename), dpi=cfg.dpi, bbox_inches="tight", transparent=True)
    plt.close()


def custom_histplot(df, column, cfg, title=None):
    """
    Create and save a custom histogram plot for a specified column in the DataFrame.

    This function creates a histogram plot using seaborn, customizes it based on the provided
    configuration, and saves it as an image file.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        column (str): The name of the column to plot.
        cfg (EDAConfig): Configuration object containing plot settings.
        title (str, optional): Custom title for the plot. If None, a default title is used.
    """
    # Apply the dark theme with transparency
    setup_dark_theme_transparent()

    plt.figure(figsize=cfg.figure_size)

    # Create histogram plot
    ax = sns.histplot(x=column, data=df, color=cfg.primary_color)

    # Customize the plot
    if title is None:
        plt.title(f"Distribution of {column}", fontsize=cfg.title_font_size, color="white")
    else:
        plt.title(title, fontsize=cfg.title_font_size, color="white")
    plt.xlabel(column, fontsize=cfg.label_font_size, color="white")
    plt.ylabel("Count", fontsize=cfg.label_font_size, color="white")

    # Save the plot with transparency
    plt.tight_layout()
    filename = f'{column}_distribution.png'
    plt.savefig(os.path.join(cfg.eda_dir, filename), dpi=cfg.dpi, bbox_inches="tight", transparent=True)
    plt.close()

def custom_wordcloud(df, column, cfg, title=None, remove_stopwords=True):
    """
    Create and save a custom word cloud for a specified column in the DataFrame.

    This function creates a word cloud using the WordCloud library, customizes it based on the
    provided configuration, and saves it as an image file. It can optionally remove stopwords.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        column (str): The name of the column to create the word cloud from.
        cfg (EDAConfig): Configuration object containing plot settings.
        title (str, optional): Custom title for the plot. If None, a default title is used.
        remove_stopwords (bool, optional): If True, remove common English stopwords. Defaults to True.
    """
    from wordcloud import WordCloud
    from collections import Counter
    import matplotlib.pyplot as plt

    # Set the style for the plot
    # sns.set_theme(style=cfg.plot_style, palette=cfg.color_palette)
    setup_dark_theme_transparent()

    plt.figure(figsize=cfg.figure_size)

    # Prepare data
    data = df[column].str.split().explode()
    if remove_stopwords:
        import nltk
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        data = [word for word in data if word.lower() not in stop_words]
    word_count = Counter(data)

    # Create wordcloud
    wc = WordCloud(width=1920, height=1080).generate_from_frequencies(word_count)

    # Customize the plot
    if title is None:
        plt.title(f"Wordcloud of {column}", fontsize=cfg.title_font_size)
    else:
        plt.title(title, fontsize=cfg.title_font_size)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    # Save the plot
    plt.tight_layout()
    filename = f'{column}_wordcloud.png'
    plt.savefig(os.path.join(cfg.eda_dir, filename), dpi=cfg.dpi, bbox_inches="tight", transparent=True)
    plt.close()

def main():
    """
    Main function to orchestrate the Exploratory Data Analysis (EDA) process.

    This function performs the following steps:
    1. Loads the configuration and sets up the font.
    2. Loads the paper data.
    3. Creates various plots (count plots, histograms, and word clouds) for different columns in the data.
    4. Saves all plots as image files in the specified EDA directory.
    """
    cfg = EDAConfig()
    setup_font(cfg)
    paper_df, paper_authors, paper_references = load_data(drop_missing=cfg.drop_missing)
    print(f"Loaded {len(paper_df)} papers!")
    print(f"Columns in df: {paper_df.columns.values}")

    os.makedirs(cfg.eda_dir, exist_ok=True)

    # count plots for different columns
    print("Creating count plots...")
    custom_countplot(paper_df, "author_count", cfg, title="Distribution of authors per paper", top_n=15)
    custom_countplot(paper_df, "reference_count", cfg, top_n=15)
    custom_countplot(paper_df, "citation_count", cfg, top_n=15)
    custom_countplot(paper_df, "update_year", cfg, top_n=15)
    custom_countplot(paper_df, "title_words", cfg, top_n=15)
    custom_countplot(paper_df, "abstract_words", cfg, top_n=15)

    # hist plots for different columns
    print("Creating hist plots...")
    custom_histplot(paper_df, "author_count", cfg)
    custom_histplot(paper_df, "reference_count", cfg)
    custom_histplot(paper_df, "citation_count", cfg)
    custom_histplot(paper_df, "update_year", cfg)
    custom_histplot(paper_df, "title_words", cfg)
    custom_histplot(paper_df, "abstract_words", cfg)

    # wordclouds for different columns
    print("Creating wordclouds...")
    custom_wordcloud(paper_df, "title", cfg)
    custom_wordcloud(paper_df, "abstract", cfg)



if __name__ == "__main__":
    main()
