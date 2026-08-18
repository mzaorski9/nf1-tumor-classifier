''' Load saved files '''

import joblib
import numpy as np
from typing import Any
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download
import streamlit as st


REPO_ID = "mzao11/tumour-risk-model"

def download_from_hf(repo_id: str, filename: str) -> str:
    """Hugging Face file downloader"""
    return hf_hub_download(repo_id, filename)

# load the function only once (protection against unnecessary function loading when re-rendering)
@st.cache_resource
def load_models() -> dict[str, Any]:
    """Load the trained RF, LR, and XGB models from disk."""
    model_rf = joblib.load(download_from_hf(REPO_ID, 'model_rf.pkl'))
    model_lr = joblib.load(download_from_hf(REPO_ID, 'model_lr.pkl'))
    model_xgb = joblib.load(download_from_hf(REPO_ID, 'model_xgb.pkl'))

    return {
        'rf': model_rf,
        'lr': model_lr,
        'xgb': model_xgb
    }

@st.cache_resource
def load_scaler() -> StandardScaler:
    """Load the fitted scaler (LR only - not needed for the rest of the models)."""
    return joblib.load(download_from_hf(REPO_ID, 'scaler_lr.pkl'))

@st.cache_data
def load_thresholds() -> dict:
    """Load the per-model decision thresholds."""
    return joblib.load(download_from_hf(REPO_ID, 'thresholds.pkl'))

@st.cache_data
def load_data() -> tuple:
    """Load the train/test feature and label splits."""
    X_te = joblib.load(download_from_hf(REPO_ID, 'X_te.pkl'))
    y_te = joblib.load(download_from_hf(REPO_ID, 'y_te.pkl'))
    X_tr = joblib.load(download_from_hf(REPO_ID, 'X_tr.pkl'))
    y_tr = joblib.load(download_from_hf(REPO_ID, 'y_tr.pkl'))
    return X_te, y_te, X_tr, y_tr

@st.cache_data
def load_y_probs() -> np.lib.npyio.NpzFile:
    """Load the saved predicted probabilities for each model."""
    with np.load(download_from_hf(REPO_ID, 'y_probs.npz')) as npz:
        return dict(npz.items())
    