class Config:
    """Configuation."""

    data_dir = r"D:\Database\arxiv"  # path to data dir
    datapath = rf"{data_dir}\sample_2024.csv"  # path to arxiv paper csv
    paper_dir = rf"{data_dir}\papers"  # path to dir where the papers will be downloaded in
    paper_metadata_path = rf"{data_dir}\papers_metadata.json"  # path to json with additional semantic scholar metadata
    multi_thread_workers = 8
    device = "cpu"  # device to use for embeddings


class EDAConfig(Config):
    """Configuration for EDA."""

    def __init__(self) -> None:
        super().__init__()

        # Existing configurations
        self.eda_dir = r"assets\eda"
        self.drop_missing = True  # drop papers without metadata from semantic scholar

        # Updated font configuration
        self.font_family = "Fira Sans"

        # Other configurations for custom countplot
        self.plot_style = "darkgrid"
        self.color_palette = "dark"
        self.font_size = 12
        self.figure_size = (12, 6)
        self.primary_color = "skyblue"
        self.title_font_size = 16
        self.label_font_size = 14
        self.x_tick_rotation = 45
        self.dpi = 300


class IndexConfig(Config):
    """Configuration for indexing."""

    def __init__(self) -> None:
        super().__init__()

        # Existing configurations
        self.vectorstore_path = r"qdrant/"
        self.embedding_model_name = "sentence-transformers/allenai-specter"
        self.text_splitter_args = {
            "chunk_size": 700,
            "chunk_overlap": 32,
            "length_function": len,
            "add_start_index": True,
        }
        self.drop_missing = True  # drop papers without metadata from semantic scholar
        self.limit = 10
        self.workers = 4
        self.collection_name = "ilyi-test"
