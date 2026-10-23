"""Original H spline-logistic and CAL Platt fitting, without file I/O."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

def as_bool(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).to_numpy(bool)
    z = s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False, "nan": False, "none": False, "": False})
    if z.isna().any(): raise ValueError("Unrecognized boolean field.")
    return z.to_numpy(bool)



def valid_labels(f: pd.DataFrame) -> np.ndarray:
    return (as_bool(f.evaluable) & as_bool(f.model_available)
            & np.isfinite(pd.to_numeric(f.R, errors="coerce"))
            & np.isfinite(pd.to_numeric(f.NSE, errors="coerce"))
            & pd.to_numeric(f.pass_joint, errors="coerce").isin([0, 1])).to_numpy()



def weights(f: pd.DataFrame) -> np.ndarray:
    """Call within one algorithm/scheme/arm, or each station x arm equally."""
    if f.empty: return np.empty(0)
    return 1.0 / f.groupby(["station_id", "arm"], observed=True).case_id.transform("size").to_numpy(float)



def design(f: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = f[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x[~np.isfinite(x)] = np.nan
    return x



def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)



def raw_probability(model: dict, frame: pd.DataFrame) -> np.ndarray:
    if model["kind"] == "constant": return np.full(len(frame), model["training_prior"])
    return model["estimator"].predict_proba(design(frame, model["columns"]))[:, 1]



def predict(model: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = raw_probability(model, frame)
    p = raw.copy()
    if model["calibrator"] is not None:
        p = model["calibrator"].predict_proba(logit(raw))[:, 1]
    available = as_bool(frame.diagnostic_available)
    raw[~available] = np.nan; p[~available] = np.nan
    return raw, p



def fit_model(train: pd.DataFrame, cal: pd.DataFrame, columns: list[str], seed: int) -> dict:
    train = train.loc[valid_labels(train) & as_bool(train.diagnostic_available)].copy()
    cal = cal.loc[valid_labels(cal) & as_bool(cal.diagnostic_available)].copy()
    if train.empty: raise ValueError("No supported and evaluable training cases.")
    w = weights(train); w *= len(w) / w.sum()
    y = train.pass_joint.astype(int).to_numpy()
    prior = float(np.average(y, weights=w))
    model = {"columns": columns, "training_prior": prior, "training_n": len(train),
             "training_stations": train.station_id.nunique(), "calibration_n": len(cal),
             "calibrator": None, "calibration_status": "identity"}
    if np.unique(y).size == 1:
        model.update(kind="constant", estimator=None)
    else:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("spline", SplineTransformer(n_knots=3, degree=2, knots="quantile", include_bias=False, extrapolation="constant")),
            ("logistic", LogisticRegression(C=1, max_iter=3000, solver="lbfgs", random_state=seed)),
        ])
        pipe.fit(design(train, columns), y, logistic__sample_weight=w)
        model.update(kind="quantile_spline_logistic", estimator=pipe)
    if len(cal) and cal.pass_joint.nunique() == 2 and model["kind"] != "constant":
        cw = weights(cal); cw *= len(cw) / cw.sum()
        platt = LogisticRegression(C=1, max_iter=3000, solver="lbfgs", random_state=seed)
        platt.fit(logit(raw_probability(model, cal)), cal.pass_joint.astype(int), sample_weight=cw)
        model.update(calibrator=platt, calibration_status="platt_fitted",
                     calibration_slope=float(platt.coef_[0, 0]), calibration_intercept=float(platt.intercept_[0]))
    model["calibration_prior"] = float(np.average(cal.pass_joint, weights=weights(cal))) if len(cal) else np.nan
    return model

