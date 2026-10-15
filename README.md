# Loss Leaders or Dead Weight? — submission files

Main report: `tafeng_loss_leaders.ipynb` (submit it together with its HTML export).

| Path | What it is |
|---|---|
| `tafeng_loss_leaders.ipynb` | Main technical report; runs top to bottom and generates every table, figure and result-driven paragraph |
| `src/data_prep.py` | Loading (both Ta Feng layouts), quality checks, cleaning, baskets |
| `src/basket_mining.py` | FP-Growth, association rules, basket-adjusted profit (BAP), pull-through, bundles, segments |
| `src/recommenders.py` | Utility matrix, popularity, repeat-buy, item-based kNN, implicit ALS |
| `src/evaluation.py` | Time-split evaluation, precision/recall/NDCG@10, profit capture, profit-aware re-ranking |
| `src/plots.py` | All figures (saved to `figures/` when the notebook runs) |
| `figures/` | PNG copies of every figure, created by the notebook |

## Running on Jojie

1. Put this folder anywhere in your Jojie workspace (keep `src/` next to the notebook).
2. The data is read in place from `/mnt/data/public/tafeng` (the four monthly files `D11`, `D12`, `D01`, `D02`); nothing is copied. `DATA_DIR` in the *Settings* cell already points there. The loader also reads the Kaggle release (`ta_feng_all_months_merged.csv`) if you point `DATA_DIR` at it, but on Jojie use the shared folder, never a home-folder path.
3. Run all cells (about 5–10 minutes; implicit ALS is the slowest part).
4. Export: `jupyter nbconvert --to html tafeng_loss_leaders.ipynb`

pyFIM (`import fim`) is used for FP-Growth when available, as in class; otherwise a pure-Python pattern-growth fallback gives the same itemsets.

No raw or processed data is included, and no file or path identifies the author.
