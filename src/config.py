class Config:
    """Configuation."""

    data_dir = r"D:\Database\arxiv"  # path to data dir
    datapath = rf"{data_dir}\sample_2017_2024.csv"  # path to arxiv paper csv
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
        self.embedding_model_name = "all-mpnet-base-v2"
        # self.qdrant_url = "https://1ed4f85b-722b-4080-97a7-afe8eab7ae7a.europe-west3-0.gcp.cloud.qdrant.io:6333"
        self.qdrant_url = "http://localhost:6333"
        # self.qdrant_path = r"D:\Database\arxiv\qdrant"
        self.text_splitter_args = [
            {
                "type": "RecursiveCharacterTextSplitter",
                "chunk_size": 1024,
                "chunk_overlap": 32,
                "length_function": len,
                "add_start_index": True,
            },
            # {
            #     "type": "SemanticChunker",
            #     "add_start_index": True,
            # },
        ]
        self.database_for = "RecursiveCharacterTextSplitter" # for which text splitter type to create the database
        self.drop_missing = True  # drop papers without metadata from semantic scholar
        self.limit = None  # limit the number of papers to index
        self.workers = 8
        self.collection_name = "arxiv_papers"
