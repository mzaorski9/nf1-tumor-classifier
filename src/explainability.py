import pandas as pd 
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import streamlit as st

""" SHAP charts """

def get_shap_values(model,
                     X: pd.DataFrame, 
                     model_type: str = "Linear",
                     scaler = None,
                     background_data: pd.DataFrame = None) \
-> shap.Explanation:
    
    """Compute SHAP values for a fitted model using either a Linear or Tree explainer.

    Parameters:
        model: A fitted model (linear or tree-based).
        X (pd.DataFrame): Data to explain.
        model_type (str): Either "Linear" or "Tree". Determines which SHAP
            explainer is used. Defaults to "Linear".
        scaler: Fitted scaler to apply before explaining (Linear models only).
            Leave as None for tree models, which are scale-invariant.
        background_data (pd.DataFrame): Background/reference data used to
            compute the explainer's baseline. If None, `X` itself is used.

    Returns:
        shap.Explanation: SHAP values for each row in `X`.

    Raises:
        TypeError: If X is not a pandas DataFrame.
        ValueError: If model_type is not "Linear" or "Tree".
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    
    if model_type not in ("Linear", "Tree"):
            raise ValueError(f"Invalid model_type: '{model_type}'. Choose 'Linear' or 'Tree'.")
    
    if scaler is not None:
        new_X = scaler.transform(X)
        df = pd.DataFrame(new_X, columns=X.columns, index=X.index)
    else:
        df = X
    
    if model_type == "Linear":
        if background_data is not None:
            background = scaler.transform(background_data) if scaler is not None else background_data
        else:
            background = df.values    
        explainer = shap.LinearExplainer(model, background)

    elif model_type == "Tree":
        # background data passed since 'interventional' peturbation (E[f(x)] calcated as the
        # average of the background data) 
        explainer = shap.TreeExplainer(
            model, 
            feature_perturbation="interventional", 
            data=background_data if background_data is not None else None
        )    
    else:
        raise ValueError("No input data provided or invalid model type: choose 'Linear' or 'Tree'.")

    return explainer(df)


def get_shap_ranking(shap_values: shap.Explanation) -> pd.DataFrame:

    """Build a feature-importance ranking table from SHAP values.

    Parameters:
        shap_values (shap.Explanation): SHAP values for a set of patients.

    Returns:
        pd.DataFrame: Two-column DataFrame ('Feature', 'Mean |SHAP|') sorted
            by descending importance.
    """

    final_shap = validate_and_extract_shap(shap_values)

    shap_df = pd.DataFrame(final_shap.values, columns=final_shap.feature_names)
    ranking = shap_df.abs().mean(axis=0).sort_values(ascending=False)

    return ranking.reset_index().rename(columns={'index': 'Feature', 0: 'Mean |SHAP|'})


def validate_and_extract_shap(shap_values: shap.Explanation) -> shap.Explanation:

    """Validate a SHAP Explanation and extract the positive-class slice if 3D."""

    if not isinstance(shap_values, shap.Explanation):
        raise ValueError("Invalid 'shap_values' argument. It must be a shap.Explanation object.")

    # Random Forest case (batch processing)
    if len(shap_values.shape) == 3:     
        return shap_values[:, :, 1]
     
    # RF (single sample processing)
    if len(shap_values.shape) == 2 and shap_values.shape[1] == 2:       
        return shap_values[:, 1]

    return shap_values

def plot_beeswarm(shap_values: shap.Explanation, model_name: str = "Model") -> plt.Figure:

    """Plots a SHAP beeswarm chart, automatically handling 2D and 3D SHAP objects.
    
    Args:
        shap_values: SHAP Explanation object containing feature importance values.
        title: Chart title (default: "Beeswarm").
    
    Returns:
        Matplotlib figure object.
    """

    final_shap = validate_and_extract_shap(shap_values)
                                            
    plt.figure(figsize=(10, 6))
    plt.title(f"Beeswarm ({model_name})", fontsize=14, pad=20)

    # 'shap' creates its own figure internally
    shap.plots.beeswarm(final_shap, show=False)
    plt.tight_layout()

    return plt.gcf()    # return current figure


def plot_bar_chart(shap_values: shap.Explanation, model_name: str = "Model") -> plt.Figure:
    """Generates a SHAP global feature importance bar chart for a given model.

    Extracts validated SHAP values and uses `shap.plots.bar` to render a 
    horizontal bar chart ranking features by their mean absolute SHAP values.

    Args:
        shap_values (shap.Explanation): A SHAP Explanation object containing 
            computed SHAP values for the dataset.
        model_name (str, optional): The display name of the model to include 
            in the plot title. Defaults to "Model".

    Returns:
        plt.Figure: The Matplotlib figure object containing the generated bar chart.
    """
    final_shap = validate_and_extract_shap(shap_values)

    plt.figure(figsize=(10, 6))
    plt.title(f"Bar chart ({model_name})", fontsize=14, pad=20)

    shap.plots.bar(final_shap, show=False)

    return plt.gcf()


def plot_waterfall(
    shap_values: shap.Explanation,
    title="Waterfall",
    clinical_mode: bool = False,
) -> plt.Figure:
    
    """Plot a SHAP waterfall chart for a specific patient.

        Automatically handles 2D/3D SHAP Explanation shapes (see
        `validate_and_extract_shap`) and, in clinical mode, converts raw
        log-odds SHAP values into probability-percentage shifts with
        doctor-friendly labels.

        Parameters:
            shap_values (shap.Explanation): SHAP values for one or more patients.
                The patient at index `pos` is the one plotted.
            title (str): Title displayed above the plot. Defaults to "Waterfall".
            clinical_mode (bool): If True, converts the base value and SHAP
                contributions from log-odds to percentage-point probability
                shifts, cleans up axis labels, and adds a baseline-vs-patient
                risk summary below the plot. If False, plots the raw SHAP
                waterfall as-is. Defaults to False.

        Returns:
            plt.figure: The matplotlib figure containing the waterfall plot.
    """

    # shap_values[0] since there's shape=(1, 18) or (1, 18, 2) for single patient
    final_shap = validate_and_extract_shap(shap_values[0]) 
    plt.figure(figsize=(10, 6))

    # convert raw values to percentages if in clinical mode 
    # note: ONLY FOR LOGISTIC REGRESSION MODEL!
    if clinical_mode:
        final_shap = final_shap.__copy__()
        
        original_base = final_shap.base_values
        original_final = original_base + final_shap.values.sum()

        prob_base = 1 / (1 + np.exp(-original_base))
        prob_final = 1 / (1 + np.exp(-original_final))

        final_shap.base_values = prob_base * 100
        
        total_odds_delta = original_final - original_base

        if abs(total_odds_delta) > 1e-5:
            total_prob_delta = (prob_final - prob_base) * 100
            final_shap.values = (final_shap.values / total_odds_delta) * total_prob_delta
        else:
            final_shap.values = np.zeros_like(final_shap.values)

        shap.plots.waterfall(final_shap, show=False)
        ax = plt.gca()

        plt.xlabel("Probability Shift [%]", fontsize=11, loc='center')

        # fix Y-Axis labels
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        cleaned_y_labels = []
        
        for lbl in y_labels:
            if "=" in lbl:
                cleaned_y_labels.append(lbl.split("=")[-1].strip())
            else:
                cleaned_y_labels.append(lbl)

        ax.set_yticklabels(cleaned_y_labels)


        xticklabels = ax.get_xticklabels()
        for lbl in xticklabels:
            txt = lbl.get_text()

            if txt == r"$f(x)$":
                lbl.set_text("")

            elif txt.startswith("$ ="):
                value = txt.replace("$", "").replace("=", "").strip()
                lbl.set_text("") 

        ax.set_xticklabels(xticklabels)

        base_value = final_shap.base_values
        final_value = base_value + final_shap.values.sum()
        
        # add summary text BELOW the plot (at figure level)
        fig = plt.gcf()
        
        fig.text(
            0.5, 0.02,  # x=0.5 (center), y=0.02 (near bottom)
            f"Baseline Population Risk: {base_value:.1f}%  |  Patient Risk: {final_value:.1f}%",
            ha='center',
            fontsize=13,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="yellow", edgecolor="black", linewidth=1.5)
        )
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # make room for the summary text below

    # Research mode
    else:
        shap.plots.waterfall(final_shap, show=False)
        fig = plt.gcf()
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()   

    return fig