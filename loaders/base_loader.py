from abc import ABC, abstractmethod
from models.page import Page

class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path : str) -> list[Page]:
        pass