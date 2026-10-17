"""
Loading, quality checks and basket building for the Ta Feng grocery data.

The raw data stays in the shared folder on Jojie and is only read from
there; nothing in this module writes a copy of it to disk.

Two layouts of Ta Feng are in circulation, and the loader accepts both:

* the original release: four files ``D11``, ``D12``, ``D01``, ``D02``
  (one per month), ``;``-separated, one header line, columns
  date; customer; age group; area; subclass; product; amount; asset; price
* the merged CSV (``ta_feng_all_months_merged.csv``) with named columns
  TRANSACTION_DT, CUSTOMER_ID, AGE_GROUP, PIN_CODE, PRODUCT_SUBCLASS,
  PRODUCT_ID, AMOUNT, ASSET, SALES_PRICE.

Both are mapped to the same tidy frame with columns
``date, customer, age, area, subclass, product, qty, cost, price``.
"""
import glob
import os

import numpy as np
import pandas as pd

# Chinese New Year 2001 fell on 24 January
CNY_2001 = pd.Timestamp("2001-01-24")

# validation and test windows used throughout the report
SPLITS = {
    "validation": {"train_end": "2001-01-01", "test_end": "2001-02-01"},
    "test": {"train_end": "2001-02-01", "test_end": "2001-03-01"},
}

# original release codes the age group with letters
AGE_CODES = {
    "A": "<25", "B": "25-29", "C": "30-34", "D": "35-39", "E": "40-44",
    "F": "45-49", "G": "50-54", "H": "55-59", "I": "60-64", "J": ">64",
    "K": "unknown",
}
AGE_ORDER = ["<25", "25-29", "30-34", "35-39", "40-44", "45-49",
             "50-54", "55-59", "60-64", ">64", "unknown"]

# folders where Ta Feng is usually mounted on Jojie (no user paths)
CANDIDATE_DIRS = [
    "/mnt/data/public/tafeng",
    "/mnt/data/public/ta-feng",
    "/mnt/data/public/ta_feng",
    "/mnt/data/public/TaFeng",
]


def find_data_dir(data_dir=None):
    """
    Locate the Ta Feng folder on Jojie.

    Parameters
    ----------
    data_dir : str, optional
        Explicit folder. If omitted, the ``TAFENG_DIR`` environment
        variable is used, then the usual public mounts are searched.

    Returns
    -------
    str
        Folder that contains the Ta Feng files, or the path of the
        merged CSV itself (e.g. ``ta_feng_all_months_merged.csv``).
    """
    tried = []
    for d in [data_dir, os.environ.get("TAFENG_DIR")] + CANDIDATE_DIRS:
        if d and (os.path.isdir(d) or os.path.isfile(d)):
            return d
        if d:
            tried.append(d)
    # last resort: anything under the public mount with "feng" in it
    hits = sorted(glob.glob("/mnt/data/public/*[Ff]eng*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        "Ta Feng folder not found. Set DATA_DIR in the notebook or the "
        f"TAFENG_DIR environment variable. Tried: {tried}"
    )


def _find_files(data_dir):
    """Return ('merged', [csv]) or ('monthly', [D11, D12, D01, D02])."""
    if os.path.isfile(data_dir):
        return "merged", [data_dir]
    files = sorted(
        f for f in glob.glob(os.path.join(data_dir, "**", "*"),
                             recursive=True)
        if os.path.isfile(f)
    )
    merged = [f for f in files
              if f.lower().endswith(".csv") and "feng" in f.lower()]
    merged.sort(key=lambda f: "all_months_merged" not in f.lower())
    if not merged:
        merged = [f for f in files if f.lower().endswith(".csv")]
    monthly = [f for f in files
               if os.path.basename(f).split(".")[0].upper()
               in {"D11", "D12", "D01", "D02"}]
    if len(monthly) >= 4:
        # order the months chronologically
        order = {"D11": 0, "D12": 1, "D01": 2, "D02": 3}
        monthly.sort(key=lambda f: order[
            os.path.basename(f).split(".")[0].upper()])
        return "monthly", monthly
    if merged:
        return "merged", merged[:1]
    raise FileNotFoundError(f"No Ta Feng files recognised in {data_dir}")


def _to_date(s):
    """
    Parse transaction dates.

    Tries, in order, the Kaggle merged-CSV format (``11/1/2000``), ISO
    (``2000-11-01``) and finally pandas' mixed-format parser.
    """
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s.str[:10], format="%Y-%m-%d", errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s, format="mixed", errors="coerce")
    return out


def _read_monthly(paths):
    """Read the original D11/D12/D01/D02 files."""
    cols = ["date", "customer", "age", "area", "subclass",
            "product", "qty", "cost", "price"]
    frames = []
    for path in paths:
        raw = pd.read_csv(
            path, sep=";", header=None, names=cols, dtype=str,
            encoding="latin-1", skipinitialspace=True,
        )
        # the header line (in Chinese) fails to parse as a date -> drop it
        raw["date"] = _to_date(raw["date"])
        frames.append(raw.dropna(subset=["date"]))
    df = pd.concat(frames, ignore_index=True)
    df["age"] = df["age"].str.strip().map(AGE_CODES).fillna("unknown")
    return df


def _read_merged(path):
    """Read the merged single-CSV release."""
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [c.strip().upper() for c in raw.columns]
    df = pd.DataFrame({
        "date": _to_date(raw["TRANSACTION_DT"]),
        "customer": raw["CUSTOMER_ID"],
        "age": raw["AGE_GROUP"],
        "area": raw["PIN_CODE"],
        "subclass": raw["PRODUCT_SUBCLASS"],
        "product": raw["PRODUCT_ID"],
        "qty": raw["AMOUNT"],
        "cost": raw["ASSET"],
        "price": raw["SALES_PRICE"],
    })
    df["age"] = (df["age"].fillna("unknown").str.strip()
                 .replace({"": "unknown", "nan": "unknown"}))
    # merged file writes letters in some versions, ranges in others
    df["age"] = df["age"].map(lambda a: AGE_CODES.get(a, a))
    # some copies write the top band as ">65" or "65+"
    df.loc[df["age"].str.startswith(">") | df["age"].str.endswith("+"),
           "age"] = ">64"
    df.loc[~df["age"].isin(AGE_ORDER), "age"] = "unknown"
    return df.dropna(subset=["date"])


def load_tafeng(data_dir=None):
    """
    Load every Ta Feng purchase line into one tidy DataFrame.

    Parameters
    ----------
    data_dir : str, optional
        Folder with the Ta Feng files (see `find_data_dir`).

    Returns
    -------
    pandas.DataFrame
        One row per purchase line with columns ``date, customer, age,
        area, subclass, product, qty, cost, price``.
    """
    kind, paths = _find_files(find_data_dir(data_dir))
    df = _read_monthly(paths) if kind == "monthly" else _read_merged(paths[0])

    for col in ["customer", "area", "subclass", "product"]:
        df[col] = df[col].astype(str).str.strip()
    # ids are zero-padded strings; strip leading zeros so both layouts agree
    df["customer"] = df["customer"].str.lstrip("0").replace("", "0")
    df["subclass"] = df["subclass"].str.lstrip("0").replace("", "0")
    for col in ["qty", "cost", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = df["date"].dt.normalize()
    df["age"] = pd.Categorical(df["age"], categories=AGE_ORDER)
    return df.reset_index(drop=True)


def quality_report(df):
    """
    Count the data-quality issues named in the game plan.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `load_tafeng`.

    Returns
    -------
    pandas.DataFrame
        One row per check with the number and share of affected lines.
    """
    n = len(df)
    checks = {
        "missing qty/cost/price": df[["qty", "cost", "price"]]
        .isna().any(axis=1),
        "qty <= 0 (returns)": df["qty"] <= 0,
        "cost <= 0": df["cost"] <= 0,
        "price <= 0": df["price"] <= 0,
        "sold below cost (price < cost)": df["price"] < df["cost"],
        "cost > 5x price (likely error)": df["cost"] > 5 * df["price"],
        "exact duplicate lines": df.duplicated(),
    }
    out = pd.DataFrame({
        "lines": {k: int(v.sum()) for k, v in checks.items()},
    })
    out["share"] = out["lines"] / n
    return out


def line_total_check(df, min_lines=50):
    """
    Test whether ``price`` and ``cost`` are line totals or unit values.

    If they are line totals, ``price / qty`` should be stable within a
    product while ``price`` itself grows with ``qty``. We compare the
    median within-product coefficient of variation (CV) of both, and the
    median within-product correlation between ``qty`` and ``price``.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `load_tafeng` (positive qty and price only are used).
    min_lines : int
        Only products with at least this many lines (and some variation
        in qty) are used.

    Returns
    -------
    pandas.DataFrame
        Median CVs and correlations for price and cost.
    """
    d = df[(df["qty"] > 0) & (df["price"] > 0) & (df["cost"] > 0)].copy()
    d["unit_price"] = d["price"] / d["qty"]
    d["unit_cost"] = d["cost"] / d["qty"]
    g = d.groupby("product")
    keep = g["qty"].transform("size").ge(min_lines) & \
        g["qty"].transform("nunique").ge(2)
    g = d[keep].groupby("product")

    def cv(s):
        return s.std() / s.mean()

    rows = {}
    for raw, unit in [("price", "unit_price"), ("cost", "unit_cost")]:
        rows[raw] = {
            "median CV of raw value": g[raw].agg(cv).median(),
            "median CV of value / qty": g[unit].agg(cv).median(),
            "median corr(qty, raw value)":
                g[["qty", raw]].corr().xs(raw, level=1)["qty"].median(),
        }
    out = pd.DataFrame(rows).T
    out.attrs["products_used"] = int(d[keep]["product"].nunique())
    return out


def clean(df, extreme_q=0.9995):
    """
    Drop lines that cannot be priced or are bulk outliers; add profit.

    Removed:

    * non-positive qty, cost or price (returns and recording errors);
    * cost more than five times the price (data-entry errors rather
      than real discounts);
    * bulk lines whose qty or price is above the `extreme_q` quantile
      (wholesale-sized purchases; a handful of them would otherwise
      dominate the per-basket averages behind BAP).

    Lines sold modestly below cost are *kept*: they are exactly the
    loss-making sales this report is about.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `load_tafeng`.
    extreme_q : float
        Quantile above which qty or price counts as a bulk outlier.

    Returns
    -------
    pandas.DataFrame
        Clean lines with an extra ``profit = price - cost`` column. The
        caps used are stored in ``out.attrs["caps"]``.
    """
    ok = (
        df[["qty", "cost", "price"]].notna().all(axis=1)
        & (df["qty"] > 0) & (df["cost"] > 0) & (df["price"] > 0)
        & (df["cost"] <= 5 * df["price"])
    )
    qty_cap = float(df.loc[ok, "qty"].quantile(extreme_q))
    price_cap = float(df.loc[ok, "price"].quantile(extreme_q))
    ok &= (df["qty"] <= qty_cap) & (df["price"] <= price_cap)
    out = df[ok].copy()
    out["profit"] = out["price"] - out["cost"]
    out.attrs["caps"] = {"qty": qty_cap, "price": price_cap}
    return out.reset_index(drop=True)


def period(df, start=None, end=None):
    """Lines with ``start <= date < end`` (either bound optional)."""
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["date"] < pd.Timestamp(end)
    return df[mask]


def make_baskets(lines):
    """
    Collapse purchase lines into baskets (one customer on one day).

    Parameters
    ----------
    lines : pandas.DataFrame
        Clean purchase lines.

    Returns
    -------
    tuple of pandas.DataFrame
        ``item_basket``: one row per (basket, subclass) with the
        subclass's revenue and profit in that basket.
        ``baskets``: one row per basket with customer, date, age,
        number of distinct subclasses (``size``), revenue and profit.
    """
    lines = lines.copy()
    lines["basket"] = lines["customer"] + "|" + \
        lines["date"].dt.strftime("%Y%m%d")
    item_basket = (
        lines.groupby(["basket", "subclass"], observed=True)
        .agg(revenue=("price", "sum"), profit=("profit", "sum"))
        .reset_index()
    )
    baskets = (
        lines.groupby("basket", observed=True)
        .agg(customer=("customer", "first"), date=("date", "first"),
             age=("age", "first"), revenue=("price", "sum"),
             profit=("profit", "sum"))
    )
    baskets["size"] = item_basket.groupby("basket").size()
    return item_basket, baskets.reset_index()


def transactions(item_basket):
    """List of subclass sets, one per basket (input for FIM)."""
    return item_basket.groupby("basket")["subclass"].apply(frozenset).tolist()
