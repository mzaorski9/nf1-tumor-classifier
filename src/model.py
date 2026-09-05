from sklearn.metrics import classification_report, RocCurveDisplay, auc, recall_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import pandas as pd
import numpy as np
from .enums import RiskLevel
from shap import Explanation
import matplotlib.pyplot as plt


''' Model-related features '''


def predict_with_threshold(X: pd.DataFrame | np.ndarray, model, threshold: float) -> tuple:
    """Generate binary predictions from a probabilistic model using a custom threshold.

    Parameters:
        X (pd.DataFrame or np.ndarray): Input feature matrix.
        model: A fitted model exposing a `predict_proba` method.
        threshold (float): Decision threshold between 0.0 and 1.0. Probabilities
            greater than or equal to this value are classified as positive (1).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of (y_pred, y_prob), where y_pred
            are the thresholded binary predictions and y_prob are the raw
            positive-class probabilities.

    Raises:
        ValueError: If threshold is out of range, or X is empty/None.
        TypeError: If the model does not support `predict_proba`.
        RuntimeError: If prediction fails due to an internal model error.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0. Received: {threshold}")

    if X is None or (isinstance(X, (pd.DataFrame, np.ndarray)) and len(X) == 0):
        raise ValueError("Input data X is empty or None.")

    if not hasattr(model, 'predict_proba'):
        raise TypeError("The passed model does not support probabilistic predictions (missing 'predict_proba').")

    try:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob

    except Exception as e:
        raise RuntimeError(f"Prediction failed due to internal model error: {e}")


def get_classification_report(y_pred, y_true) -> dict:
    """Build a classification report comparing predicted vs. true labels.

    Parameters:
        y_pred (array-like): Predicted binary labels.
        y_true (array-like): Ground-truth binary labels.

    Returns:
        dict: A scikit-learn classification report (precision, recall, f1-score,
            support) keyed by class name ("No Tumour", "Tumour") and averages.

    Raises:
        ValueError: If either array is empty, or if their lengths mismatch.
    """
    if len(y_pred) < 1 or len(y_true) < 1:
        raise ValueError("Cannot generate classification report with empty arrays.")

    if len(y_pred) != len(y_true):
        raise ValueError(f"Shape mismatch: y_pred has length {len(y_pred)}, but y_true has length {len(y_true)}.")

    return classification_report(
        y_true,
        y_pred,
        target_names=["No Tumour", "Tumour"],
        output_dict=True
    )


def get_risk_level(probability, threshold):
    """Return a risk label based on a probability and threshold.

    Parameters:
        probability (int, float, numpy.ndarray): A single probability value or array of probability values.
        threshold (float): The decision threshold between 0.0 and 1.0.

    Returns:
        str or numpy.ndarray: "High tumor risk!" or "Low tumor risk" for each input probability.

    Raises:
        ValueError: If the threshold is not between 0.0 and 1.0.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0. Received: {threshold}")

    thresh_f32 = np.float32(threshold)

    # single patient case
    if isinstance(probability, (int, float)) or (hasattr(probability, 'size') and probability.size == 1):
        prob = probability.item() if hasattr(probability, 'item') else float(probability)
        if prob >= thresh_f32:
            risk = RiskLevel.HIGH
        elif prob >= 0.8 * thresh_f32:
            risk = RiskLevel.MED
        else:
            risk = RiskLevel.LOW
        return risk.value

    # multi patient case
    return np.where(
        probability >= thresh_f32,
        RiskLevel.HIGH.value,
        np.where(
            probability >= 0.8 * thresh_f32,
            RiskLevel.MED.value,
            RiskLevel.LOW.value
        )
    )


def get_pred_contributors(shap_vals: Explanation, patient: pd.DataFrame, n: int = 3):
    """Extract the top risk and protective SHAP contributors for a single patient.

    Parameters:
        shap_vals (shap.Explanation): SHAP values for the patient(s); only the
            first row (index 0) is used.
        patient (pd.DataFrame): Patient feature data; column order must match
            the SHAP values.
        n (int): Maximum number of risk/protective factors to return for each
            category. Defaults to 3.

    Returns:
        tuple[list, list]: A tuple of (risk_factors, protect_factors), each a
            list of (shap_value, feature_name) pairs sorted by ascending
            absolute SHAP magnitude (most important last).

    Raises:
        ValueError: If shap_vals or patient is None, if n is negative, or if
            the number of SHAP values does not match the number of features.
        AttributeError: If shap_vals lacks a 'values' attribute, or patient
            lacks a 'columns' attribute.
        RuntimeError: If contributor extraction fails for any other reason.
    """
    if shap_vals is None or patient is None:
        raise ValueError("shap_vals and patient cannot be None.")

    if not hasattr(shap_vals, 'values'):
        raise AttributeError("shap_vals does not have 'values' attribute.")

    if not hasattr(patient, 'columns'):
        raise AttributeError("patient does not have 'columns' attribute.")

    if n < 0:
        raise ValueError(f"Parameter n must be non-negative. Received: {n}")

    try:
        shap = shap_vals.values[0]
        features = patient.columns.tolist()

        if len(shap) != len(features):
            raise ValueError(f"Shape mismatch: shap has length {len(shap)}, but features has length {len(features)}.")

        pairs = [(s, f) for s, f in zip(shap, features)]
        pairs = sorted(pairs, key=lambda x: abs(x[0]))

        risk_factors = [(s, f) for s, f in pairs if s > 0.05]
        protect_factors = [(s, f) for s, f in pairs if s < -0.05]

        return risk_factors[-n:], protect_factors[-n:]

    except Exception as e:
        raise RuntimeError(f"Error reading prediction contributors: {e}")


def generate_result_comment(risk_factors, protect_factors, risk_lvl: RiskLevel, patient_data):
    """Build a human-readable Markdown comment summarizing the key drivers of a prediction.

    Parameters:
        risk_factors (list): List of (shap_value, feature_name) pairs pushing
            the prediction toward higher risk.
        protect_factors (list): List of (shap_value, feature_name) pairs
            pushing the prediction toward lower risk.
        risk_lvl (RiskLevel): The overall risk level for the patient
            (HIGH, MED, or LOW).
        patient_data (pd.DataFrame): Single-row patient feature data, used to
            display the raw value alongside each factor.

    Returns:
        str: A Markdown-formatted summary listing the main risk or protective
            factors, or a fallback message if no single feature dominates.
    """
    result = []

    # Check against the .value attribute if you chose to return strings
    # (or leave as is if you chose the 'class RiskLevel(str, Enum)' approach)
    if risk_lvl == RiskLevel.HIGH.value or risk_lvl == RiskLevel.MED.value:
        # Markdown bullet points, most important factor listed first
        factors = '\n'.join([f"- {format_feature_names(f)} ({patient_data.iloc[0][f]})" for s, f in reversed(risk_factors)])
        result.append(f"🔴 **Main risk factors (from the most important):**\n{factors}")

    elif risk_lvl == RiskLevel.LOW.value:
        # Markdown bullet points, most important factor listed first
        factors = '\n'.join([f"- {format_feature_names(f)} ({patient_data.iloc[0][f]})" for s, f in reversed(protect_factors)])
        result.append(f"🟢 **Main protective factors (from the most important):**\n{factors}")

    else:
        result.append("No single feature strongly drives this prediction — "
                       "the result reflects a combination of mild signals.")

    # Join sections with double newlines so Markdown treats them as separate paragraphs
    return '\n\n'.join(result)


def format_feature_names(name: str) -> str:
    """Convert a raw feature column name into a display-friendly label.

    Parameters:
        name (str): Raw feature name (e.g. with underscores or internal codes).

    Returns:
        str: Human-readable feature name with underscores replaced by spaces
            and known abbreviations expanded.
    """
    return name.replace('_', ' ').replace('Café au lait CLS', 'Café au lait spots')


def find_optimal_threshold(recalls: np.ndarray, precisions: np.ndarray, thresholds: np.ndarray) -> float:
    """Find the optimal probability threshold prioritizing maximum patient safety (Recall),
    then selecting the highest Precision available at that maximum Recall.

    Parameters:
        recalls (np.ndarray): Recall values, typically from
            `sklearn.metrics.precision_recall_curve`.
        precisions (np.ndarray): Precision values aligned with `recalls`.
        thresholds (np.ndarray): Threshold values aligned with `recalls` and
            `precisions`.

    Returns:
        float: The threshold that achieves maximum recall with the highest
            precision among ties.

    Note:
        Ensure arrays are aligned (e.g., sliced with [:-1]) before passing
        from scikit-learn's precision_recall_curve.
    """
    # find the maximum possible recall (usually 1.0)
    recall_max = recalls.max()

    # mask precisions where recall isn't maximized (keeps array length unchanged)
    masked_precisions = np.where(recalls == recall_max, precisions, -1)

    # find the original index of the highest precision
    best_idx = np.argmax(masked_precisions)

    return thresholds[best_idx]


def get_features_importance(
    pipeline: Pipeline,
    feature_names: list,
    xgb_importance_type: str = 'gain'
) -> pd.DataFrame:
    """Extract and normalize feature importances from a fitted sklearn Pipeline.

    Supports Logistic Regression (coefficients), XGBoost (booster importance
    scores), and Random Forest / other tree-based models (feature_importances_).

    Parameters:
        pipeline (Pipeline): A fitted sklearn Pipeline containing a step named
            'model'.
        feature_names (list): Ordered list of feature names matching the
            model's input columns.
        xgb_importance_type (str): Importance type to request from XGBoost's
            booster (e.g. 'gain', 'weight', 'cover'). Defaults to 'gain'.

    Returns:
        pd.DataFrame: A two-column DataFrame ('Feature', 'Importance') sorted
            by descending importance.

    Raises:
        TypeError: If `pipeline` is not a sklearn Pipeline, or if the model
            type is unsupported.
        ValueError: If the pipeline has no 'model' step, or if the number of
            importance values does not match the number of features.
    """
    if not hasattr(pipeline, 'named_steps'):
        raise TypeError("Expected a sklearn Pipeline.")

    model = pipeline.named_steps.get('model')

    if model is None:
        raise ValueError("Pipeline does not contain a 'model' step.")

    result_dict = {}
    params_arr = []

    # Logistic Regression
    if hasattr(model, 'coef_'):
        params_arr = model.coef_[0]

    # XGBoost
    elif hasattr(model, 'get_booster'):
        scores_dict = model.get_booster().get_score(
            importance_type=xgb_importance_type
        )

        for i, feat in enumerate(feature_names):
            result_dict[feat] = scores_dict.get(
                feat,
                scores_dict.get(f"f{i}", 0.0)
            )

    # Random Forest / tree-based models
    elif hasattr(model, 'feature_importances_'):
        params_arr = model.feature_importances_

    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

    # LR / RF
    if len(params_arr) > 0:
        if len(params_arr) != len(feature_names):
            raise ValueError(
                f"Number of importance values ({len(params_arr)}) "
                f"does not match number of features ({len(feature_names)})."
            )

        for score, feat in zip(params_arr, feature_names):
            result_dict[feat] = score

    result_df = pd.DataFrame(
        result_dict,
        index=['Importance']
    ).T

    result_df.index.name = 'Feature'
    result_df = result_df.reset_index()

    return result_df.sort_values(
        by='Importance',
        ascending=False
    ).reset_index(drop=True)


def generate_roc_cv_fold_plot(estimator, 
                              cv, 
                              X, 
                              y, 
                              threshold: float = 0.5, 
                              model_name: str = "Model"
) -> tuple[plt.Figure, pd.DataFrame, dict[str, float]]:
    """
    Generates an ROC curve per CV fold alongside a fold scores table and summary statistics.
    
    Parameters:
    - estimator: Tuned Pipeline or model object (or fitted GridSearchCV).
    - cv: Cross-validation splitter object (e.g., StratifiedKFold).
    - X, y: Features and target (Pandas DataFrame/Series or NumPy arrays).
    - threshold (float): Decision threshold for binary classification predictions.
    - model_name (str): Name of the model to display in the plot title.

    Returns:
    - fig (plt.Figure): Matplotlib figure of the ROC curves.
    - scores_fold_df (pd.DataFrame): DataFrame containing per-fold AUC and Recall scores.
    - summary (dict): Dictionary containing mean and std for AUC and Recall.
    """
    # Handle both GridSearchCV objects and standard Pipeline/Estimator objects
    if hasattr(estimator, "best_estimator_"):
        base_model = estimator.best_estimator_
    else:
        base_model = estimator


    # Convert to numpy arrays if necessary to support generic slicing
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else X
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y

    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs, recalls = [], [], []
    fold_scores = []

    # re-run the 5 folds manually on the BEST model configuration to plot the lines;
    # treat it as 5 separate datasets;
    # NOTE: GridSearchCV do the same loop under the hood with cv.split(X_train2, y_train2)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr)):
        # resets weights with each loop       
        init_model = clone(base_model)
        X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
        y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # fit (search for weights) and plot curve for this fold
        init_model.fit(X_fold_train, y_fold_train)

        viz = RocCurveDisplay.from_estimator(
            init_model, X_fold_test, y_fold_test,
            name=f"ROC fold {fold}", alpha=0.3, lw=1, ax=ax
        )
        
        # FPR/TPR array sizes can differ between folds because they depend on
        # the number of unique prediction scores (thresholds) produced by predict_proba();
        # intepr() interpolates values according to np.linspace(0, 1, 100) and produces 5 arrays of the same size
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr) 
        aucs.append(viz.roc_auc)

        y_prob_fold = init_model.predict_proba(X_fold_test)[:, 1]
        y_pred_fold = (y_prob_fold >= threshold).astype(int)
        recall_fold = recall_score(y_fold_test, y_pred_fold)
        recalls.append(recall_fold)

        fold_scores.append({
            "Fold": f"Fold {fold}",
            "AUC": viz.roc_auc,
            "Recall": recall_fold
        })

    # add the Mean ROC line
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # AUC for the mean ROC
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr, mean_tpr, color="b",
        label=f"Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})", lw=2, alpha=0.8
    )

    ax.plot([0, 1], [0, 1], "r--", label="Chance")
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"ROC per Fold (Best {model_name} Model)")
    ax.legend(loc="lower right")

    # generate dataframe of scores per fold
    scores_fold_df = pd.DataFrame(fold_scores)

    # summary dict (means)
    summary = {
        "mean_auc": np.mean(aucs),
        "mean_recall": np.mean(recalls),
        "std_auc": np.std(aucs),
        "std_recall": np.std(recalls)
    }

    return (fig, scores_fold_df, summary)

