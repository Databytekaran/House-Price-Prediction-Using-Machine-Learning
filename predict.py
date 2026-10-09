"""Predict property prices with the saved model.

Usage
-----
    python predict.py test.csv submission.csv
    python predict.py test.csv submission.csv --sample-submission sample_submission.csv

* test.csv must have the same columns as the training file (without the target PropPrice).
  It must contain the PropertyID column; other missing fields are tolerated (the pipeline
  treats them as unknown), but a warning is printed.
* --sample-submission (optional) makes the output use that file's column names and row order,
  and stops with an error if its IDs do not match the IDs in test.csv.

property_utils.py and property_price_model.pkl must be in the same folder as this script
(the pickle needs property_utils.py to be importable).
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Known harmless message: the saved pipeline contains an all-empty PropertyID column that the imputer skips
# (its coefficient in the model is 0). Hidden here so the command-line output stays readable.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from property_utils import ID_COL, RAW_FEATURES, TARGET  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_csv", help="CSV with the property features (e.g. test.csv)")
    ap.add_argument("output_csv", help="where to write the predictions (e.g. submission.csv)")
    ap.add_argument("--model", default=str(HERE / "property_price_model.pkl"), help="path to the saved model")
    ap.add_argument("--sample-submission", default=None, help="optional sample_submission.csv to copy the format from")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if ID_COL not in df.columns:
        sys.exit(f"ERROR: '{ID_COL}' column not found in {args.input_csv}")
    if df[ID_COL].duplicated().any():
        sys.exit(f"ERROR: duplicate {ID_COL} values in {args.input_csv}")
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} expected column(s) missing, treated as unknown: {missing}")

    model = joblib.load(args.model)
    pred = model.predict(df.drop(columns=[TARGET], errors="ignore"))

    if not np.isfinite(pred).all() or (pred <= 0).any():
        sys.exit("ERROR: model produced non-finite or non-positive prices - check the input file")

    sub = pd.DataFrame({ID_COL: df[ID_COL].values, TARGET: np.round(pred).astype(int)})

    if args.sample_submission:
        ss = pd.read_csv(args.sample_submission)
        if ss.shape[1] != 2:
            sys.exit(f"ERROR: expected 2 columns in {args.sample_submission}, found {list(ss.columns)}")
        id_name = ss.columns[0]
        if ss[id_name].duplicated().any() or set(ss[id_name]) != set(sub[ID_COL]):
            sys.exit(f"ERROR: IDs in {args.sample_submission} ({len(ss)} rows) do not match IDs in "
                     f"{args.input_csv} ({len(sub)} rows)")
        sub = sub.set_index(ID_COL).loc[ss[id_name]].reset_index()   # same row order as the sample file
        sub.columns = list(ss.columns)                              # same column names

    sub.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(sub)} predictions to {args.output_csv} | min = {sub.iloc[:, 1].min():,} | max = {sub.iloc[:, 1].max():,}")


if __name__ == "__main__":
    main()
