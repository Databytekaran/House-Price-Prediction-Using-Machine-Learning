"""Helper code for the Property Price Prediction capstone project.
All functions here work ROW-BY-ROW (or learn only from training data inside a
Pipeline), so they cannot leak information from the validation/test data."""
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "PropPrice"
ID_COL = "PropertyID"

RAW_FEATURES = ['PropertyID', 'PropertyClass', 'PropertyZone', 'PropertyFrontage', 'PropertySize', 'Street', 'Alley', 'PropertyShape', 'Elevation', 'Amenities', 'Orientation', 'Grade', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'PropertyStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Roof1Material', 'Roof2Material', 'ExteriorCladdingType', 'ExteriorCladdingArea', 'ExterQual', 'ExterCond', 'PropertyFooting', 'BsmntFinish', 'BsmntMaintenance', 'BsmntVisibility', 'BsmntFinRat1', 'BsmntFinSty1', 'BsmntFinQual1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmntSqFtage', 'Heating', 'HeatingEfficiency', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'Bath1', 'Bath2', 'BedroomUpLev', 'KitchenUpLev', 'KitchenQual', 'CntRmsUpLev', 'Functional', 'CntFireplaces', 'QualFireplace', 'BasementType', 'BasementYrBlt', 'BasementFinish', 'BasementCars', 'BasementSqFootage', 'BasementQual', 'BasementCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'BoundaryFeatures', 'AddFeatures', 'AddVal', 'SaleMon', 'SaleYr', 'SaleType', 'SaleCondn']

# ------------------------------------------------------------------
# 1. ORDINAL columns  ->  explicit order (lowest -> highest)
# ------------------------------------------------------------------
QUAL = ["None", "Po", "Fa", "TA", "Gd", "Ex"]          # Poor < Fair < Typical < Good < Excellent
ORDINAL_ORDER = {
    "ExterQual": QUAL, "ExterCond": QUAL, "HeatingEfficiency": QUAL, "KitchenQual": QUAL,
    "QualFireplace": QUAL, "PoolQC": QUAL,
    "BsmntFinish": QUAL, "BsmntMaintenance": QUAL,           # basement height / condition
    "BasementQual": QUAL, "BasementCond": QUAL,              # (these 'Basement*' columns hold GARAGE data)
    "BsmntVisibility": ["None", "No", "Mn", "Av", "Gd"],     # basement exposure
    "BsmntFinRat1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmntFinQual1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BasementFinish": ["None", "Unf", "RFn", "Fin"],         # garage interior finish
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"],
    "PropertyShape": ["IR3", "IR2", "IR1", "Reg"],
    "Grade": ["Sev", "Mod", "Gtl"],                           # land slope
    "Amenities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],      # utilities
    "Street": ["Grvl", "Pave"],
    "CentralAir": ["N", "Y"],
    "BoundaryFeatures": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],   # fence quality / privacy
}
ORDINAL_COLS = list(ORDINAL_ORDER)

# ------------------------------------------------------------------
# 2. NOMINAL columns  ->  One-Hot Encoding (no natural order)
# ------------------------------------------------------------------
NOMINAL_COLS = [
    "PropertyClass", "PropertyZone", "Alley", "Elevation", "Orientation", "Neighborhood",
    "Condition1", "Condition2", "BldgType", "PropertyStyle", "RoofStyle", "RoofMatl",
    "Roof1Material", "Roof2Material", "ExteriorCladdingType", "PropertyFooting", "Heating",
    "Electrical", "BasementType", "AddFeatures", "SaleType", "SaleCondn", "SaleMon",
]
# PropertyClass is a numeric CODE (20, 60, ...) and SaleMon is a month number -> treated as categories.

# ------------------------------------------------------------------
# 3. Cleaning + feature engineering (row-wise, leakage-free)
# ------------------------------------------------------------------
def clean_data(df):
    """Convert 'structural absence' (e.g. no pool, no garage) from NaN to an explicit 'None'
    category using fillna(). Real unknown values stay NaN and are imputed later inside the
    Pipeline (fitted on training data only)."""
    df = df.copy()
    df = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns])
    df = df.replace("NA", np.nan)
    for c in RAW_FEATURES:                       # allow partial inputs (missing fields become NaN)
        if c not in df.columns:
            df[c] = np.nan

    # --- basement columns: NaN + zero basement area  ->  'None' ---
    no_bsmt = df["BsmntSqFtage"].fillna(0) == 0
    for c in ["BsmntFinish", "BsmntMaintenance", "BsmntVisibility", "BsmntFinRat1", "BsmntFinQual1"]:
        df[c] = df[c].where(~(no_bsmt & df[c].isna()), "None")
    # --- 'Basement*' columns actually describe the GARAGE: NaN + zero cars -> 'None' ---
    no_garage = df["BasementCars"].fillna(0) == 0
    for c in ["BasementType", "BasementFinish", "BasementQual", "BasementCond"]:
        df[c] = df[c].where(~(no_garage & df[c].isna()), "None")
    df["BasementYrBlt"] = df["BasementYrBlt"].fillna(df["YearBuilt"])     # no garage -> use house year
    # --- fireplace / pool / alley / fence / misc: NaN really means 'feature not present' ---
    df["QualFireplace"] = df["QualFireplace"].where(~((df["CntFireplaces"] == 0) & df["QualFireplace"].isna()), "None")
    df["PoolQC"] = df["PoolQC"].where(~((df["PoolArea"] == 0) & df["PoolQC"].isna()), "None")
    for c in ["Alley", "BoundaryFeatures", "AddFeatures"]:
        df[c] = df[c].fillna("None")
    # --- masonry veneer: pandas reads the category 'None' as NaN; if the area is known it is 'None' ---
    df["ExteriorCladdingType"] = df["ExteriorCladdingType"].where(
        ~(df["ExteriorCladdingType"].isna() & df["ExteriorCladdingArea"].notna()), "None")
    # --- codes that are really categories ---
    df["PropertyClass"] = df["PropertyClass"].astype(str)
    df["SaleMon"] = df["SaleMon"].astype(str)
    return df


def add_features(df):
    """Domain-knowledge features (each uses only the same row)."""
    df = df.copy()
    df["TotalSF"] = df["BsmntSqFtage"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = df["Bath1"] + 0.5 * df["Bath2"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    df["HouseAge"] = (df["SaleYr"] - df["YearBuilt"]).clip(lower=0)
    df["RemodelAge"] = (df["SaleYr"] - df["YearRemodAdd"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["HasGarage"] = (df["BasementCars"] > 0).astype(int)
    df["HasBasement"] = (df["BsmntSqFtage"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasFireplace"] = (df["CntFireplaces"] > 0).astype(int)
    df["QualxLivArea"] = df["OverallQual"] * df["GrLivArea"]
    return df


def prepare_df(df, engineer=True):
    out = clean_data(df)
    return add_features(out) if engineer else out


# ------------------------------------------------------------------
# 4. Small custom transformer: log1p on skewed numeric columns (skew learnt on TRAIN only)
# ------------------------------------------------------------------
class SkewFixer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        sk = np.apply_along_axis(lambda c: skew(c), 0, X)
        self.cols_ = np.where((np.abs(sk) > self.threshold) & (X.min(axis=0) >= 0))[0]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[:, self.cols_] = np.log1p(X[:, self.cols_])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ------------------------------------------------------------------
# 5. Pipeline builders
# ------------------------------------------------------------------
def build_encoder(scaled=True):
    """ColumnTransformer: numeric / ordinal / nominal handled separately.
    scaled=True  -> for linear models & PCA (log-skew fix + StandardScaler)
    scaled=False -> for tree models (scaling is unnecessary)"""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scaled:
        num_steps += [("skew", SkewFixer()), ("scale", StandardScaler())]
    numeric = Pipeline(num_steps)

    ordinal_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[ORDINAL_ORDER[c] for c in ORDINAL_COLS],
                                  handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
    if scaled:
        ordinal_steps.append(("scale", StandardScaler()))
    ordinal = Pipeline(ordinal_steps)

    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("ord", ordinal, ORDINAL_COLS),
        ("nom", nominal, NOMINAL_COLS),
    ], verbose_feature_names_out=False)


def make_model(estimator, scaled=True, engineer=True, pca=None, log_target=True):
    steps = [
        ("prep", FunctionTransformer(prepare_df, kw_args={"engineer": engineer})),
        ("encode", build_encoder(scaled)),
    ]
    if pca is not None:
        steps.append(("pca", PCA(n_components=pca, random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return TransformedTargetRegressor(regressor=pipe)   # identity transform


def predict_property_price(model, property_data):
    """property_data: dict (one property) or DataFrame (many properties).
    Missing fields are allowed (NaN) and handled by the pipeline.
    Returns a float for one property (dict) and a NumPy array for a DataFrame."""
    if isinstance(property_data, dict):
        return float(model.predict(pd.DataFrame([property_data]))[0])
    return model.predict(property_data.copy())


# ------------------------------------------------------------------
# 6. Evaluation metrics (safe when a true price is 0)
# ------------------------------------------------------------------
def regression_metrics(y_true, y_pred, margins=(10, 20)):
    """R2, MAE, RMSE and percentage-error metrics.

    A percentage error divides by the true price, which is undefined when the true
    price is 0. Such rows are EXCLUDED from MAPE, median abs % error and the
    'within X %' shares (and counted in 'zero-target rows excluded'), so no
    division-by-zero or inf ever appears. R2, MAE and RMSE use all rows.
    (The PropPrice column has no zero values, so nothing is excluded in this project.)"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {"R2": r2_score(y_true, y_pred),
           "MAE": mean_absolute_error(y_true, y_pred),
           "RMSE": root_mean_squared_error(y_true, y_pred)}
    ok = y_true != 0
    ape = np.abs((y_true[ok] - y_pred[ok]) / y_true[ok]) * 100 if ok.any() else np.array([])
    out["MAPE %"] = float(ape.mean()) if ape.size else float("nan")
    out["Median abs % error"] = float(np.median(ape)) if ape.size else float("nan")
    for m in margins:
        out[f"Within {m}% of real price"] = float((ape <= m).mean() * 100) if ape.size else float("nan")
    out["Zero-target rows excluded"] = int((~ok).sum())
    return out
