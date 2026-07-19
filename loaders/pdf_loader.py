from pathlib import Path
import pymupdf

from models.page import Page

project_root = Path(__file__).resolve().parent.parent
folder_path = project_root / "data" / "raw"

def load_pdf_to_pages() -> list[Page]:
    pdf_files = list(folder_path.glob("*.pdf"))
    pdf_pages=[]
    for file_path in pdf_files:
        document = pymupdf.open(file_path);
        for page_num, page in enumerate(document):
            text = page.get_text("text");
            page_data = Page(file_path.name, page_num+1, text)
            pdf_pages.append(page_data)

        document.close()
    return pdf_pages