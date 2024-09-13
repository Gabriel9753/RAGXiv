"""
"""

class Config:
    """ Configuation. """

    datapath = r"data\genai\Owners_Manual_tesla.pdf"
    text_splitter_args = {
        "chunk_size": 300,
        "chunk_overlap": 100,
        "length_function": len,
        "add_start_index": True,
    }
    chromapath = r"chroma/"