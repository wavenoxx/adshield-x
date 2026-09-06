"""
AdShield-X :: models.py
-----------------------
The classical ML zoo reproduced from the base paper, plus a single evaluation
harness so every model is scored identically.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             average_precision_score, matthews_corrcoef)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_ml_models(seed=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, C=1.0, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=4,
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=18,
                                                n_jobs=-1, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=220,
                                   random_state=seed, early_stopping=True),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=180,
                                                        max_depth=4, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=350, learning_rate=0.06,
                                   num_leaves=48, random_state=seed, n_jobs=-1,
                                   verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=350, learning_rate=0.08, max_depth=6,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="logloss", random_state=seed,
                                 n_jobs=-1, tree_method="hist"),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel="rbf", C=2.0, probability=True, random_state=seed),
    }


def evaluate(name, model, X_te, y_te, thr=0.5, train_time=None):
    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)
        proba = proba[:, 1] if proba.ndim == 2 else proba
    else:
        proba = model.decision_function(X_te)
    infer_time = time.perf_counter() - t0
    pred = (proba >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return {
        "Algorithm": name,
        "Accuracy": accuracy_score(y_te, pred) * 100,
        "Precision": precision_score(y_te, pred, zero_division=0) * 100,
        "Recall": recall_score(y_te, pred, zero_division=0) * 100,
        "F1-Score": f1_score(y_te, pred, zero_division=0) * 100,
        "ROC-AUC": roc_auc_score(y_te, proba) * 100,
        "PR-AUC": average_precision_score(y_te, proba) * 100,
        "MCC": matthews_corrcoef(y_te, pred),
        "FPR": fp / max(1, (fp + tn)) * 100,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "TrainTime_s": round(train_time, 2) if train_time else None,
        "InferTime_ms_per_1k": round(infer_time / len(X_te) * 1e6, 3),
    }


def results_table(rows):
    df = pd.DataFrame(rows)
    num = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "FPR"]
    for c in num:
        df[c] = df[c].round(2)
    df["MCC"] = df["MCC"].round(4)
    return df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
