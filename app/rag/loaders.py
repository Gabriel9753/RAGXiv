from typing import AsyncIterator, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import fitz  # PyMuPDF


class CustomPDFLoader(BaseLoader):
    """A custom document loader that reads a PDF and extracts text, images, and metadata."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the PDF file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that reads a PDF file and extracts text, images, and metadata page by page."""
        doc = fitz.open(self.file_path)  # Open the PDF file

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Get the page

            # Extract text
            text = page.get_text("text")

            # Extract images
            images = []
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]  # image XREF
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                images.append({"index": img_index, "image_bytes": image_bytes, "image_ext": image_ext})

            # Extract metadata
            metadata = {
                "source": self.file_path,
                "page_number": page_num + 1,
                "image_count": len(images),
                "total_pages": len(doc),
            }

            # Yield text content and associated metadata as a Document
            if text:
                yield Document(page_content=text, metadata=metadata)

            # Yield each image as a separate Document
            for image in images:
                yield Document(
                    page_content=f"Image {image['index']}",
                    metadata={
                        **metadata,
                        "image_extension": image["image_ext"],
                        "image_data": image["image_bytes"],
                    },
                )

        doc.close()

    async def alazy_load(self) -> AsyncIterator[Document]:
        """An async lazy loader that reads a PDF file and extracts text, images, and metadata asynchronously."""
        # import aiofiles

        doc = fitz.open(self.file_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Get the page

            # Extract text
            text = page.get_text("text")

            # Extract images
            images = []
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]  # image XREF
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                images.append({"index": img_index, "image_bytes": image_bytes, "image_ext": image_ext})

            # Extract metadata
            metadata = {
                "source": self.file_path,
                "page_number": page_num + 1,
                "image_count": len(images),
                "total_pages": len(doc),
            }

            # Yield text content and associated metadata as a Document
            if text:
                yield Document(page_content=text, metadata=metadata)

            # Yield each image as a separate Document
            for image in images:
                yield Document(
                    page_content=f"Image {image['index']}",
                    metadata={
                        **metadata,
                        "image_extension": image["image_ext"],
                        "image_data": image["image_bytes"],
                    },
                )

        doc.close()
