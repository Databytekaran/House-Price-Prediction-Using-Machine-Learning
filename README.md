# Property Price Prediction (Capstone Project 1)

Predicts `PropPrice` (sale price) of a property with a Lasso regression pipeline (log-price target).

## Files
| File | Purpose |
|---|---|
| `Property_Price_Prediction.ipynb` | Full workflow: EDA, cleaning, encoding, feature engineering, models, tuning, evaluation |
| `Project_Summary_Report.docx` | Written report |
| `property_utils.py` | Helper module (cleaning, features, pipeline builders, `regression_metrics`, `predict_property_price`) |
| `property_price_model.pkl` | Final saved model (trained on all labelled rows except 2 non-representative Partial sales) |
| `predict.py` | Command-line prediction for any CSV that has the training columns |
| `submission.csv` | Predictions for the **219-property internal hold-out set** (see limitations) |
| `requirements.txt` | Package versions |

## Run from a clean environment
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
1. **Notebook:** put `Digicrome_project_dataset.csv` next to the notebook, open it and choose *Restart & Run All*.
2. **Predict a new file:**
   ```
   python predict.py test.csv submission.csv
   python predict.py test.csv submission.csv --sample-submission sample_submission.csv   # copy its format exactly
   ```
3. **Predict in Python:**
   ```python
   import joblib
   from property_utils import predict_property_price     # property_utils.py must be importable
   model = joblib.load("property_price_model.pkl")
   predict_property_price(model, {"OverallQual": 7, "GrLivArea": 1800, "Neighborhood": "CollgCr"})   # missing fields are allowed
   ```

## Limitations (read before submitting)
* `test.csv`, `sample_submission.csv` and `data_description.txt` were **not available**. Therefore `submission.csv` is **not** an official test-set submission: it holds predictions for the internal 15 % hold-out split, in the format `PropertyID, PropPrice`. Feature meanings come from the project PDF's metadata and could not be checked against a separate description file.
* The dataset file was not available when the project was last reviewed, so the notebook was **not re-executed** then and the metrics in the notebook and report are the values recorded in the notebook's saved outputs. Run all cells once to refresh them.
* Known harmless quirk: `clean_data()` drops `PropertyID` and then re-creates it as an empty column, so the pipeline contains one constant missing-indicator column (`missingindicator_PropertyID`, Lasso coefficient 0.0). It does not change predictions. Removing it would change the pipeline and require retraining the model.
