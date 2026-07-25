import faiss

class FAISSFactory:

    @staticmethod
    def create_flat_l2(dimension: int):
        return faiss.IndexFlatL2(dimension)

    @staticmethod
    def create_flat_ip(dimension: int):
        return faiss.IndexFlatIP(dimension)

    @staticmethod
    def create_hnsw(dimension: int):
        return faiss.IndexHNSWFlat(dimension, 32)