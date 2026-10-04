# Bank Marketing Response Prediction

Predict whether a client subscribes to a term deposit after a bank marketing call.
Compares a dummy, a linear (logistic regression) and a tree (decision tree) baseline,
plus a random forest as an extra model. The decision threshold is chosen on the
validation set only. Evaluation focus: PR AUC, positive-class recall, threshold trade-off.

## Data
Moro, Rita & Cortez (2014), *Bank Marketing* (bank-additional-full.csv), UCI Machine Learning Repository.
1. Download from https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Unzip and place `bank-additional-full.csv` in `data/` (semicolon-separated).

The data are not redistributed in this repo. Check the UCI page for the licence terms (CC BY 4.0 at time of writing; please verify).

## Run
```
pip install -r requirements.txt
python run_project.py --data data/bank-additional-full.csv
```
Outputs go to `outputs/`: `results.csv`, `confusion_test.csv`, `run_info.json`, and figures
(class balance, test PR curves, validation threshold trade-off). Runs in a few minutes on a laptop CPU.
`--synthetic` runs a smoke test on fake data and must not be used for reported results.

## Method summary
- Target `y` (yes=1). `duration` is dropped because it is only known after the call (leakage).
- `pdays == 999` is converted to a `previously_contacted` flag.
- Stratified 60/20/20 train/validation/test split, seed 42. No customer ID exists, so group splitting is not possible (state this as a limitation).
- Hyperparameters tuned on validation PR AUC. Threshold: highest validation precision with recall >= 0.60.
- Test set used once for the final table.

## AI assistance
Code scaffold drafted with Claude (Anthropic) and then reviewed and run by the author. See the report's AI statement.
