import math
import numpy as np
import torch
from scipy.stats import norm, rankdata


class SCSOT:
    def __init__(
        self,
        epsilon=0.05,
        max_iter=200,
        tol=1e-6,
        chunk_size=1024,
        rank_method="average",
        device=None,
        dtype=torch.float32,
    ):
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.chunk_size = int(chunk_size)
        self.rank_method = rank_method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    def copula_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        Z = np.empty((n, d), dtype=np.float64)

        for j in range(d):
            ranks = rankdata(X[:, j], method=self.rank_method)
            u = ranks / (n + 1.0)
            Z[:, j] = norm.ppf(u)

        return Z

    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=self.dtype, device=self.device)

    def _squared_euclidean(self, X, Y):
        x2 = (X * X).sum(dim=1, keepdim=True)
        y2 = (Y * Y).sum(dim=1).unsqueeze(0)
        return (x2 + y2 - 2.0 * X @ Y.T).clamp_min(0.0)

    def _row_logsumexp(self, X, Y, g):
        values = []

        for start in range(0, X.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, X.shape[0])
            cost = self._squared_euclidean(X[start:stop], Y)
            values.append(
                torch.logsumexp(
                    (g.unsqueeze(0) - cost) / self.epsilon,
                    dim=1,
                )
            )

        return torch.cat(values, dim=0)

    def _col_logsumexp(self, X, Y, f):
        values = []

        for start in range(0, Y.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, Y.shape[0])
            cost = self._squared_euclidean(X, Y[start:stop])
            values.append(
                torch.logsumexp(
                    (f.unsqueeze(1) - cost) / self.epsilon,
                    dim=0,
                )
            )

        return torch.cat(values, dim=0)

    def entropic_ot(self, X, Y):
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        n = X.shape[0]
        m = Y.shape[0]

        log_a = torch.full(
            (n,),
            -math.log(n),
            dtype=self.dtype,
            device=self.device,
        )
        log_b = torch.full(
            (m,),
            -math.log(m),
            dtype=self.dtype,
            device=self.device,
        )

        f = torch.zeros(n, dtype=self.dtype, device=self.device)
        g = torch.zeros(m, dtype=self.dtype, device=self.device)

        for _ in range(self.max_iter):
            f_old = f.clone()
            g_old = g.clone()

            f = self.epsilon * (
                log_a - self._row_logsumexp(X, Y, g)
            )
            g = self.epsilon * (
                log_b - self._col_logsumexp(X, Y, f)
            )

            delta = max(
                torch.max(torch.abs(f - f_old)).item(),
                torch.max(torch.abs(g - g_old)).item(),
            )

            if delta < self.tol:
                break

        a = torch.full(
            (n,),
            1.0 / n,
            dtype=self.dtype,
            device=self.device,
        )
        b = torch.full(
            (m,),
            1.0 / m,
            dtype=self.dtype,
            device=self.device,
        )

        return torch.dot(a, f) + torch.dot(b, g) - self.epsilon

    def sinkhorn_divergence(self, source, target):
        source_z = self.copula_transform(source)
        target_z = self.copula_transform(target)

        cross = self.entropic_ot(source_z, target_z)
        source_self = self.entropic_ot(source_z, source_z)
        target_self = self.entropic_ot(target_z, target_z)

        return (
            cross
            - 0.5 * source_self
            - 0.5 * target_self
        ).item()

    def select_source(self, candidate_sources, target):
        target_z = self.copula_transform(target)
        target_self = self.entropic_ot(target_z, target_z)

        scores = []

        for source in candidate_sources:
            source_z = self.copula_transform(source)
            cross = self.entropic_ot(source_z, target_z)
            source_self = self.entropic_ot(source_z, source_z)

            score = (
                cross
                - 0.5 * source_self
                - 0.5 * target_self
            ).item()

            scores.append(score)

        selected_index = int(np.argmin(scores))

        return selected_index, np.asarray(scores, dtype=np.float64)
