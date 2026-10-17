"""
Offline evaluation and profit-aware re-ranking.

Evaluation is a time split: models are trained on one window and scored
on the purchases in the next one. For every evaluated customer we take
the top-10 list and compute precision@10, recall@10, NDCG@10 and the
*actual* profit the store made in the test window on recommended
subclasses that the customer bought.

Two modes are reported:

* ``all``: any subclass may be recommended (buy-again counts);
* ``new``: subclasses the customer bought in training are removed from
  the candidates and from the ground truth (discovery only).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_REC = 10
N_CAND = 50
LAMBDAS = np.round(np.linspace(0, 1, 11), 2)


class TestSet:
    """
    Ground truth for one evaluation window.

    Parameters
    ----------
    R : scipy.sparse.csr_matrix
        Training utility matrix (rows = `users`, columns = `items`).
    users, items : pandas.Index
        Row and column labels of `R`.
    test_lines : pandas.DataFrame
        Clean purchase lines in the test window.
    min_train_baskets : int
        Customers need at least this many training baskets.
    train_baskets : pandas.Series
        Number of training baskets per customer.
    """

    def __init__(self, R, users, items, test_lines, train_baskets,
                 min_train_baskets=3):
        t = test_lines[test_lines["subclass"].isin(items)
                       & test_lines["customer"].isin(users)]
        keep = train_baskets.reindex(users).fillna(0) >= min_train_baskets
        eligible = users[keep.to_numpy()]
        t = t[t["customer"].isin(eligible)]
        prof = t.groupby(["customer", "subclass"], observed=True)["profit"].sum()

        self.customers = pd.Index(sorted(prof.index
                                         .get_level_values(0).unique()))
        self.rows = users.get_indexer(self.customers)
        r = self.customers.get_indexer(prof.index.get_level_values(0))
        c = items.get_indexer(prof.index.get_level_values(1))
        shape = (len(self.customers), len(items))
        self.truth = np.zeros(shape, dtype=bool)
        self.truth[r, c] = True
        self.profit = np.zeros(shape, dtype=np.float32)
        self.profit[r, c] = prof.to_numpy()
        self.seen = R[self.rows].toarray() > 0
        self.n_items = len(items)

        # basket view of the test window: which subclasses each basket has
        t = t.assign(basket=t["customer"] + "|" + t["date"].dt.strftime("%Y%m%d"))
        bp = t.groupby("basket")["profit"].sum()
        self.basket_profit = bp.to_numpy()
        self.basket_user = self.customers.get_indexer(
            t.groupby("basket")["customer"].first().reindex(bp.index))
        self.basket_items = np.zeros((len(bp), len(items)), dtype=bool)
        self.basket_items[bp.index.get_indexer(t["basket"]),
                          items.get_indexer(t["subclass"])] = True


def candidates(model, test, mode="all", n_cand=N_CAND, batch=2048):
    """
    Top `n_cand` items per evaluated customer, best first.

    Parameters
    ----------
    model : fitted recommender
        Anything with ``score(rows)``.
    test : TestSet
        Evaluation window.
    mode : {"all", "new"}
        ``new`` removes items bought in training.

    Returns
    -------
    numpy.ndarray
        ``n_customers x n_cand`` array of item column indices.
    """
    out = []
    for s in range(0, len(test.rows), batch):
        sc = np.asarray(model.score(test.rows[s:s + batch]), dtype=float)
        if mode == "new":
            sc[test.seen[s:s + batch]] = -np.inf
        top = np.argpartition(-sc, n_cand, axis=1)[:, :n_cand]
        order = np.argsort(-np.take_along_axis(sc, top, axis=1),
                           axis=1, kind="stable")
        out.append(np.take_along_axis(top, order, axis=1))
    return np.vstack(out)


def rerank(cand, value, lam, n=N_REC):
    """
    Profit-aware re-ranking of each customer's candidate list.

    Both relevance and value are turned into rank percentiles *within*
    the customer's candidates (1 = best), then blended:

    .. math:: s = (1-\\lambda)\\,pct(\\text{relevance}) +
              \\lambda\\,pct(\\text{value})

    Parameters
    ----------
    cand : numpy.ndarray
        Candidates from `candidates`, best first.
    value : numpy.ndarray
        One value per item column (e.g. BAP or own margin).
    lam : float
        Weight on value, from 0 (pure relevance) to 1 (pure value).

    Returns
    -------
    numpy.ndarray
        ``n_customers x n`` re-ranked item indices.
    """
    k = cand.shape[1]
    rel_pct = np.tile((k - np.arange(k)) / k, (len(cand), 1))
    if lam == 0:
        return cand[:, :n]
    val_pct = rankdata(value[cand], axis=1) / k
    score = (1 - lam) * rel_pct + lam * val_pct
    order = np.argsort(-score, axis=1, kind="stable")[:, :n]
    return np.take_along_axis(cand, order, axis=1)


def metrics(recs, test, mode="all"):
    """
    Precision@n, recall@n, NDCG@n, profit and coverage of a top-n list.

    NDCG uses binary relevance, i.e. the class DCG with
    :math:`2^{r}-1 = 1` for a bought item and 0 otherwise.

    Two profit numbers are reported, both from *actual* test-window
    purchases (never from BAP, to avoid circularity):

    * ``profit_per_customer``: profit on the recommended subclasses the
      customer bought (the measure in the game plan);
    * ``basket_profit_per_customer``: total profit of every test basket
      that contains at least one of those subclasses, i.e. the whole
      basket a successful recommendation sat in.

    Returns
    -------
    dict
        Averages over customers who have at least one relevant item.
    """
    truth = test.truth & ~test.seen if mode == "new" else test.truth
    n_rel = truth.sum(axis=1)
    ok = n_rel > 0
    hits = np.take_along_axis(truth, recs, axis=1)[ok]
    prof = np.take_along_axis(test.profit, recs, axis=1)[ok]
    n = recs.shape[1]
    disc = 1 / np.log2(np.arange(2, n + 2))
    dcg = (hits * disc).sum(axis=1)
    idcg = np.array([disc[:min(m, n)].sum() for m in n_rel[ok]])
    # baskets of evaluated customers that contain a hit item
    rec_ok = np.take_along_axis(truth, recs, axis=1)
    in_eval = ok[test.basket_user]
    b_recs = recs[test.basket_user]
    b_hit = (np.take_along_axis(test.basket_items, b_recs, axis=1)
             & rec_ok[test.basket_user]).any(axis=1) & in_eval
    basket_prof = test.basket_profit[b_hit].sum() / ok.sum()
    return {
        "customers": int(ok.sum()),
        "precision@10": hits.sum(axis=1).mean() / n,
        "recall@10": (hits.sum(axis=1) / n_rel[ok]).mean(),
        "ndcg@10": (dcg / idcg).mean(),
        # profit on recommended items the customer actually bought
        "profit_per_customer": (prof * hits).sum(axis=1).mean(),
        "basket_profit_per_customer": basket_prof,
        "coverage": len(np.unique(recs[ok])) / test.n_items,
    }


def evaluate_models(models, test, modes=("all", "new")):
    """Metrics table for several fitted models (no re-ranking)."""
    rows = []
    for m in models:
        for mode in modes:
            cand = candidates(m, test, mode)
            rows.append({"model": m.name, "mode": mode,
                         **metrics(cand[:, :N_REC], test, mode)})
    return pd.DataFrame(rows)


def lambda_sweep(cand, test, values, mode="all", lambdas=LAMBDAS):
    """
    Accuracy and profit for every lambda and every value signal.

    Parameters
    ----------
    cand : numpy.ndarray
        Candidate lists from `candidates`.
    test : TestSet
        Evaluation window.
    values : dict
        ``{signal name: value per item column}``, e.g. BAP and margin.

    Returns
    -------
    pandas.DataFrame
        One row per (signal, lambda).
    """
    rows = []
    for name, v in values.items():
        for lam in lambdas:
            recs = rerank(cand, v, lam)
            rows.append({"signal": name, "lambda": lam,
                         **metrics(recs, test, mode)})
    return pd.DataFrame(rows)


def pick_lambda(sweep, signal="BAP", max_ndcg_loss=0.05):
    """
    Lambda with the highest profit that keeps NDCG@10 within
    `max_ndcg_loss` (relative) of the pure-relevance list.
    """
    s = sweep[sweep["signal"] == signal]
    base = s.loc[s["lambda"] == 0, "ndcg@10"].iloc[0]
    ok = s[s["ndcg@10"] >= (1 - max_ndcg_loss) * base]
    return float(ok.sort_values(["profit_per_customer", "lambda"],
                                ascending=[False, True])["lambda"].iloc[0])


def item_values(bap, items, col):
    """Map a BAP-table column onto item columns; unscored items get the median."""
    v = bap[col].reindex(items)
    return v.fillna(bap[col].median()).to_numpy(dtype=float)


def per_customer_ndcg(recs, test, mode="all"):
    """
    NDCG@n for every evaluated customer (same definition as `metrics`).

    Returns
    -------
    pandas.Series
        Indexed by customer ID; customers with no relevant item are
        left out, as in `metrics`.
    """
    truth = test.truth & ~test.seen if mode == "new" else test.truth
    n_rel = truth.sum(axis=1)
    ok = n_rel > 0
    hits = np.take_along_axis(truth, recs, axis=1)[ok]
    n = recs.shape[1]
    disc = 1 / np.log2(np.arange(2, n + 2))
    idcg = np.array([disc[:min(m, n)].sum() for m in n_rel[ok]])
    return pd.Series((hits * disc).sum(axis=1) / idcg,
                     index=test.customers[ok])


ACTIVITY_BINS = [1, 3, 6, 12, np.inf]
ACTIVITY_LABELS = ["2-3", "4-6", "7-12", "13+"]


def ndcg_by_activity(models, test, train_baskets, modes=("all", "new"),
                     bins=ACTIVITY_BINS, labels=ACTIVITY_LABELS):
    """
    Mean NDCG@10 by how many training baskets a customer has.

    Shows *who* each model serves well: light shoppers (little history)
    or regulars (long history).

    Parameters
    ----------
    models : list
        Fitted recommenders.
    test : TestSet
        Evaluation window.
    train_baskets : pandas.Series
        Number of training baskets per customer.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, mode, activity, ndcg@10, customers``.
    """
    bucket = pd.cut(train_baskets.reindex(test.customers),
                    bins=bins, labels=labels)
    rows = []
    for m in models:
        for mode in modes:
            if mode == "new" and m.name == "Repeat-buy":
                continue  # identical to popularity once history is removed
            nd = per_customer_ndcg(candidates(m, test, mode)[:, :N_REC],
                                   test, mode)
            g = nd.groupby(bucket.reindex(nd.index), observed=False)
            for lab, vals in g:
                rows.append({"model": m.name, "mode": mode,
                             "activity": lab, "ndcg@10": vals.mean(),
                             "customers": len(vals)})
    out = pd.DataFrame(rows)
    out["activity"] = pd.Categorical(out["activity"], categories=labels,
                                     ordered=True)
    return out
