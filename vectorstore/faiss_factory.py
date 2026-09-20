import faiss


class FAISSFactory:

    @staticmethod
    def create_flat_l2(
        dimension: int
    ) -> faiss.Index:
        return faiss.IndexFlatL2(dimension)

    @staticmethod
    def create_flat_ip(
        dimension: int
    ) -> faiss.Index:
        return faiss.IndexFlatIP(dimension)

    @staticmethod
    def create_hnsw(
        dimension: int
    ) -> faiss.Index:
        return faiss.IndexHNSWFlat(
            dimension,
            32
        )