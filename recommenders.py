"""
Implicit-feedback recommenders for the Ta Feng customer x subclass matrix.

All models share one interface:

* ``fit(R)`` with ``R`` a users x items ``scipy.sparse.csr_matrix`` of
  implicit ratings ``r_ui = log(1 + distinct days u bought i)``;
* ``score(rows)`` returning a dense ``len(rows) x n_items`` array where a
  higher value means a better recommendation.

Models: global popularity, repeat-buy (the customer's own history),
item-based neighbourhood CF, and implicit alternating least squares.
"""
import numpy as np
import pandas as pd
from scipy import sparse


def build_utility(train_lines, users=None):
    """
    Build the implicit utility matrix from training purchase lines.

    Parameters
    ----------
    train_lines : pandas.DataFrame
        Clean purchase lines from the training window.
    users : iterable, optional
        Customers to keep as rows (default: everyone in `train_lines`).

    Returns
    -------
    tuple
        ``(R, users, items)``: CSR matrix with
        ``log(1 + distinct purchase days)`` and the row / column labels
        as ``pandas.Index``.
    """
    days = (train_lines.groupby(["customer", "subclass"], observed=True)
            ["date"].nunique())
    items = pd.Index(sorted(days.index.get_level_values(1).unique()))
    if users is None:
        users = days.index.get_level_values(0).unique()
    users = pd.Index(sorted(users))
    days = days[days.index.get_level_values(0).isin(users)]
    r = users.get_indexer(days.index.get_level_values(0))
    c = items.get_indexer(days.index.get_level_values(1))
    R = sparse.csr_matrix(
        (np.log1p(days.to_numpy(dtype=float)), (r, c)),
        shape=(len(users), len(items)),
    )
    return R, users, items


class Popularity:
    """Rank items by how many customers bought them (same list for all)."""

    name = "Popularity"

    def fit(self, R):
        self.pop = np.asarray((R > 0).sum(axis=0)).ravel().astype(float)
        return self

    def score(self, rows):
        return np.broadcast_to(self.pop, (len(rows), len(self.pop))).copy()


class RepeatBuy:
    """
    Rank items by the customer's own implicit rating (buy-again list).

    Not collaborative: included because grocery shoppers repeat a lot,
    so any CF model should be judged against it. Popularity breaks ties
    and fills the list for items the customer never bought.
    """

    name = "Repeat-buy"

    def fit(self, R):
        self.R = R
        pop = np.asarray((R > 0).sum(axis=0)).ravel().astype(float)
        self.tiebreak = 1e-6 * pop / pop.max()
        return self

    def score(self, rows):
        return self.R[rows].toarray() + self.tiebreak


class ItemKNN:
    """
    Item-based neighbourhood CF on implicit ratings.

    Similarity is cosine between item columns of ``R``. Because only
    positive feedback is observed (an empty cell means "did not buy",
    not "unknown"), the matrix is not mean-centred: centring would make
    every bought item look identical. Each item keeps its `k` most
    similar items, and the score of item :math:`i` for user :math:`u` is

    .. math:: \\hat r_{ui} = \\sum_{j \\in N_k(i) \\cap I_u} sim(i,j)\\, r_{uj}

    i.e. the class item-CF formula without the division by the sum of
    similarities. With positive-only data that division would turn the
    score into an average of the user's ratings and give the top spot
    to items backed by a single weak neighbour; the plain sum ranks by
    total evidence instead.
    """

    def __init__(self, k=50):
        self.k = k
        self.name = f"Item-kNN (k={k})"

    def fit(self, R):
        X = R.tocsc().astype(float)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        S = (X.T @ X).toarray() / np.outer(norms, norms)
        np.fill_diagonal(S, 0.0)
        # keep the k largest similarities in each column (neighbours of i)
        k = min(self.k, S.shape[0] - 1)
        drop = np.argpartition(-S, k, axis=0)[k:, :]
        np.put_along_axis(S, drop, 0.0, axis=0)
        self.S = sparse.csr_matrix(S)
        self.R = R
        return self

    def score(self, rows):
        return np.asarray((self.R[rows] @ self.S).todense())


class ImplicitALS:
    """
    Alternating least squares for implicit feedback (Hu, Koren and
    Volinsky, 2008).

    Every cell gets a preference :math:`p_{ui} = 1` if bought else 0 and
    a confidence :math:`c_{ui} = 1 + \\alpha r_{ui}`. ALS alternates the
    two ridge-regression steps from class, now weighted by confidence:

    .. math:: x_u = (Y^TY + Y^T(C_u - I)Y + \\lambda I)^{-1} Y^T C_u p_u

    and the same for item factors with the roles swapped. The
    :math:`Y^TY` term is shared by all users, so each solve only touches
    the items the user bought.
    """

    def __init__(self, factors=32, reg=0.1, alpha=10.0, iters=12, seed=0):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iters, self.seed = iters, seed
        self.name = f"Implicit ALS (d={factors}, alpha={alpha:g})"

    def _half_step(self, Cm, Y):
        """Solve every row of the other factor matrix with Y fixed."""
        d = Y.shape[1]
        YtY = Y.T @ Y + self.reg * np.eye(d)
        X = np.zeros((Cm.shape[0], d))
        indptr, indices, data = Cm.indptr, Cm.indices, Cm.data
        for u in range(Cm.shape[0]):
            lo, hi = indptr[u], indptr[u + 1]
            if lo == hi:
                continue
            Yu, cu = Y[indices[lo:hi]], data[lo:hi]  # cu = c_ui - 1
            A = YtY + (Yu.T * cu) @ Yu
            b = Yu.T @ (1.0 + cu)  # Y^T C_u p_u, p = 1 on bought items
            X[u] = np.linalg.solve(A, b)
        return X

    def fit(self, R):
        rng = np.random.default_rng(self.seed)
        Cu = (R * self.alpha).tocsr()
        Ci = Cu.T.tocsr()
        # small random start (all-ones would keep every factor identical)
        self.Y = rng.normal(0, 0.01, (R.shape[1], self.factors))
        self.X = np.zeros((R.shape[0], self.factors))
        self.loss = []
        for _ in range(self.iters):
            self.X = self._half_step(Cu, self.Y)
            self.Y = self._half_step(Ci, self.X)
            self.loss.append(self._observed_rmse(R))
        return self

    def _observed_rmse(self, R):
        """RMSE of predicted preference on bought cells (convergence check)."""
        coo = R.tocoo()
        pred = np.einsum("ij,ij->i", self.X[coo.row], self.Y[coo.col])
        return float(np.sqrt(np.mean((1 - pred) ** 2)))

    def score(self, rows):
        return self.X[rows] @ self.Y.T


class AssocRuleRec:
    """
    Recommender built from FP-Growth pair rules (bridges FIM and CF).

    For every pair rule :math:`j \\rightarrow i` with lift > 1, the
    confidence is stored in a matrix :math:`C_{ji}`. A customer who
    bought the set :math:`I_u` fires every rule whose antecedent is in
    :math:`I_u`, and item :math:`i` is scored by the strongest rule
    pointing to it (the classic rule-based recommender in Aggarwal,
    *Recommender Systems*, ch. 3):

    .. math:: \\hat r_{ui} = \\max_{j \\in I_u} C_{ji}

    Ties are broken by the sum of fired confidences, then popularity.

    Parameters
    ----------
    rules : pandas.DataFrame
        Output of `basket_mining.assoc`, mined on the *training* baskets.
    items : pandas.Index
        Column labels of the utility matrix.
    """

    def __init__(self, rules, items):
        self.rules, self.items = rules, items
        self.name = "FP-Growth rules"

    def fit(self, R):
        r = self.rules[(self.rules["n_ante"] == 1) & (self.rules["lift"] > 1)]
        j = self.items.get_indexer(r["antecedent"].str[0])
        i = self.items.get_indexer(r["consequent"])
        keep = (j >= 0) & (i >= 0)
        m = len(self.items)
        self.C = sparse.csr_matrix(
            (r["confidence"].to_numpy()[keep], (j[keep], i[keep])),
            shape=(m, m))
        self.n_rules = int(keep.sum())
        self.B = (R > 0).astype(float).tocsr()
        pop = np.asarray(self.B.sum(axis=0)).ravel()
        self.tiebreak = 1e-6 * pop / max(pop.max(), 1)
        return self

    def score(self, rows):
        B = self.B[rows]
        total = np.asarray((B @ self.C).todense())       # sum of fired rules
        best = np.zeros_like(total)
        for u in range(B.shape[0]):
            fired = B.indices[B.indptr[u]:B.indptr[u + 1]]
            if len(fired):
                best[u] = self.C[fired].max(axis=0).toarray().ravel()
        return best + 1e-3 * total + self.tiebreak
