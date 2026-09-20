from pathlib import Path

import pymupdf

from loaders.base_loader import BaseLoader
from models.page import Page


class PDFLoader(BaseLoader):

    def load(
        self,
        file_path: Path
    ) -> list[Page]:

        if not file_path:
            raise ValueError(
                "File path cannot be empty."
            )

        pdf_files = list(
            file_path.glob("*.pdf")
        )

        pdf_pages = []

        for pdf_file in pdf_files:

            document = pymupdf.open(pdf_file)

            for page_num, page in enumerate(
                document
            ):
                text = page.get_text("text")

                page_data = Page(
                    pdf_file.name,
                    page_num + 1,
                    text
                )

                pdf_pages.append(page_data)

            document.close()

        return pdf_pages