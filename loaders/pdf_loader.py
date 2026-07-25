from pathlib import Path
import pymupdf
from loaders.base_loader import BaseLoader
from models.page import Page

class PDFLoader(BaseLoader):
    def load(self, file_path : str) -> list[Page]:
        if not file_path:
            ValueError("File Path given is empty")

        pdf_files = list(file_path.glob("*.pdf"))
        pdf_pages=[]

        for file_path in pdf_files:
            document = pymupdf.open(file_path);
            for page_num, page in enumerate(document):
                text = page.get_text("text");
                page_data = Page(file_path.name, page_num+1, text)
                pdf_pages.append(page_data)
            document.close()
        return pdf_pages