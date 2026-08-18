import pandas as pd 
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

""" SHAP charts """

def get_shap_values(model,
                     scaler, 
                     X: pd.DataFrame, 
                     model_type="Linear", 
                     background_data: pd.DataFrame = None) \
-> shap.Explanation:

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    scaled_array = scaler.transform(X)
    scaled_df = pd.DataFrame(scaled_array, columns=X.columns, index=X.index)

    if model_type == "Linear" and background_data is not None:
        explainer = shap.LinearExplainer(model, background_data)

    elif model_type == "Tree":
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional", data=scaled_df)
    
    else:
        raise ValueError("No input data provided or invalid model type: choose 'Linear' or 'Tree'.")

    shap_vals = explainer(scaled_df)
    
    return shap_vals


def plot_beeswarm(shap_values: shap.Explanation, title="Beeswarm"):
    """Plots a SHAP beeswarm chart, automatically handling 2D and 3D SHAP objects.
    
    Args:
        shap_values: SHAP Explanation object containing feature importance values.
        title: Chart title (default: "Beeswarm").
    
    Returns:
        Matplotlib figure object.
    """

    if not isinstance(shap_values, shap.Explanation):
        raise ValueError("Invalid 'shap_values' argument. It must be shap.Explanation object.")
    
    if len(shap_values.shape) == 3:
        final_shap = shap_values[:, :, 1]
    else:
        final_shap = shap_values
    
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14, pad=20)

    shap.plots.beeswarm(final_shap, show=False)

    plt.tight_layout()
    return plt.gcf()



def plot_waterfall(
    shap_values: shap.Explanation,
    pos: int = 0,
    title="Waterfall",
    clinical_mode: bool = False,
) -> plt.figure:
    
    """Plots a SHAP waterfall chart for a specific patient index (pos),

    automatically handling 2D/3D objects and formatting labels cleanly for doctors.
    """

    if not isinstance(pos, int):
        raise ValueError("Invalid 'position' argument. It must be an integer.")

    if not isinstance(shap_values, shap.Explanation):
        raise ValueError("Invalid 'shap_values' argument. It must be a shap.Explanation object.")

    if len(shap_values.shape) == 3:
        final_shap = shap_values[pos, :, 1]
    else:
        final_shap = shap_values[pos]

    # convert raw values to percentages if in clinical mode
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

        plt.figure(figsize=(10, 6))
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
        
        return fig