# src/metrics_predictors.py
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def as_2d(x):
    x = np.asarray(x)
    return x.reshape(x.shape[0], -1)


def _make_clf(kind: str, seed: int = 0):
    kind = str(kind).lower()

    if kind in ["logreg", "lr", "logistic"]:
        return LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
            class_weight="balanced",
        )

    if kind in ["knn", "k-nn"]:
        return KNeighborsClassifier(n_neighbors=15, weights="distance")

    if kind in ["rf", "random_forest"]:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )

    if kind in ["mlp", "nn"]:
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            max_iter=400,
            random_state=seed,
            early_stopping=True,
        )

    raise ValueError(f"Unknown classifier kind={kind}")



def fit_classifier(train_X, train_y, seed=0, kind="logreg", scale=True):
    y = np.asarray(train_y).ravel()

    # If numeric labels look integer-like, cast to int (nice for stability)
    if np.issubdtype(y.dtype, np.number):
        if np.all(np.isfinite(y)) and np.allclose(y, np.round(y)):
            y = y.astype(int)

    X = as_2d(train_X)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    clf = _make_clf(kind=kind, seed=seed)
    clf.fit(X, y)
    return scaler, clf



def predict_classifier(scaler, clf, X):
    X = as_2d(X)
    if scaler is not None:
        X = scaler.transform(X)

    # prefer proba if available
    if hasattr(clf, "predict_proba"):
        p1 = clf.predict_proba(X)[:, 1]
        yhat = (p1 >= 0.5).astype(int)
        return yhat, p1
    else:
        yhat = clf.predict(X)
        return yhat, None


def binary_metrics(train_X, train_y, test_X, test_y, seed=0, kind="logreg", scale=True):
    ytr = np.asarray(train_y).astype(int).ravel()
    yte = np.asarray(test_y).astype(int).ravel()

    scaler, clf = fit_classifier(train_X, ytr, seed=seed, kind=kind, scale=scale)

    yhat_tr, p1_tr = predict_classifier(scaler, clf, train_X)
    yhat_te, p1_te = predict_classifier(scaler, clf, test_X)

    acc_tr = accuracy_score(ytr, yhat_tr)
    acc_te = accuracy_score(yte, yhat_te)

    # AUC only if we have probabilities
    auc_tr = roc_auc_score(ytr, p1_tr) if p1_tr is not None else np.nan
    auc_te = roc_auc_score(yte, p1_te) if p1_te is not None else np.nan

    return float(auc_tr), float(acc_tr), float(auc_te), float(acc_te)


def cond_pred_train_test(train_X, train_y, test_X, test_y, seed=0, prefix="cond_from_X",
                         kind="logreg", scale=True):
    auc_tr, acc_tr, auc_te, acc_te = binary_metrics(
        train_X, train_y, test_X, test_y, seed=seed, kind=kind, scale=scale
    )
    return {
        f"{prefix}_auc": {"train": auc_tr, "test": auc_te},
        f"{prefix}_acc": {"train": acc_tr, "test": acc_te},
    }


def cluster_pred_train_test_mcc(train_X, train_labels, test_X, test_labels,
                               seed=0, kind="knn", scale=True):
    """
    Multi-class label prediction with MCC.
    """
    ytr = np.asarray(train_labels).ravel()
    yte = np.asarray(test_labels).ravel()

    # Need >=2 classes on both
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"mcc": {"train": 0.0, "test": 0.0}}

    scaler, clf = fit_classifier(train_X, ytr, seed=seed, kind=kind, scale=scale)

    Xtr = as_2d(train_X); Xte = as_2d(test_X)
    if scaler is not None:
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

    pred_tr = clf.predict(Xtr)
    pred_te = clf.predict(Xte)

    mcc_tr = matthews_corrcoef(ytr, pred_tr)
    mcc_te = matthews_corrcoef(yte, pred_te)
    return {"mcc": {"train": float(mcc_tr), "test": float(mcc_te)}}
