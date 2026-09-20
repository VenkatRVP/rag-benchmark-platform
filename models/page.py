class Page:

    def __init__(
        self,
        document_name: str,
        page_number: int,
        text: str
    ) -> None:
        self.document_name = document_name
        self.page_number = page_number
        self.text = text