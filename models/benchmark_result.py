class BenchmarkResult:

    def __init__(
        self,
        recall: float = 0.0,
        precision: float = 0.0,
        hit_rate: float = 0.0,
        mrr: float = 0.0
    ) -> None:
        self.recall = recall
        self.precision = precision
        self.hit_rate = hit_rate
        self.mrr = mrr