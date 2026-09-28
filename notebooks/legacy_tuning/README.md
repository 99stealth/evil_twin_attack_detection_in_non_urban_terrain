# Legacy tuning notebooks (exploratory)

These 18 notebooks (9 algorithms x Base/FE) are kept unchanged for transparency and
history. They use `train_test_split(test_size=0.25, random_state=42, stratify=y)`
for the train/test split and `GridSearchCV(cv=5)` for hyperparameter search — both
a random row-level shuffle.

The dataset is not i.i.d. rows: it consists of 48 attack sessions of ~11-19 rows
each (a ~55s window resampled onto a 5s grid), so a shuffled split or a shuffled
K-fold puts near-identical rows of the same session into both train and test. The
model can then partly recognise the session instead of generalising, which
inflates the reported accuracy (most visibly for Point/Antenna prediction).

**These notebooks are exploratory and superseded by
[`../02_final_comparison.ipynb`](../02_final_comparison.ipynb)**, which uses the
session-aware, grouped cross-validation protocol defined in
[`../../src/eval_protocol.py`](../../src/eval_protocol.py) and is the source of
every number and figure reported in the manuscript. See the repository
[README](../../README.md#evaluation-protocol) for the protocol description and
[`../03_protocol_comparison.ipynb`](../03_protocol_comparison.ipynb) for a direct
side-by-side of the shuffled/legacy/grouped protocols on the same models.
