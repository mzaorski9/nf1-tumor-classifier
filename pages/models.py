
from src.data_loader import load_data, load_models, load_thresholds, load_y_probs
from src.model import get_features_importance, generate_roc_cv_fold_plot
import streamlit as st
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns


models     = load_models()      # dict
y_probs    = load_y_probs()
thresholds = load_thresholds()  # dict
X_te, y_te, X_tr, y_tr      = load_data()


feature_names = [
    'Case_Type', 'Age_of_Mother', 'Age_of_Father', 'Age_at_First_Diagnosis',
    'Café_au_lait_CLS', 'Axillary_Freckles', 'Inguinal_Freckles', 'Lisch_Nodules',
    'Dermal_Neurofibromins', 'Plexiform_Neurofibromins', 'Optic_Glioma',
    'Skeletal_Dysplasia', 'Learning_Disability', 'Hypertension', 'Astrocytoma',
    'Hamartoma', 'Scoliosis', 'Other_Symptoms'
]

def make_markdown_table(df):
    header = '| Model | ' + ' | '.join(df.columns) + ' |'
    separator = '|---|' + '---|' * len(df.columns)
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = f"{row[col]:.3f}"
            if row[col] == df[col].max():
                val = f"**{val} ✅**"
            cells.append(val)
        rows.append(f"| {idx} | " + " | ".join(cells) + " |")
    return '\n'.join([header, separator] + rows)



st.subheader("Models Analysis")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("##### Models Comparison Table")

scores = {}
predicts = {}
mapped_names = {'lr': 'Logistic Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
inv_names = { v : k for k, v in mapped_names.items()}

for name, model in models.items():
    
    thresh = thresholds[name]
    new_name = mapped_names[name]

    y_prob = y_probs[name]
    y_pred_opt = (y_prob >= thresh).astype(int)

    class_report = classification_report(
        y_te, 
        y_pred_opt,
        target_names=['No Tumour', 'Tumour'], 
        output_dict=True
    ) 

    scores[new_name] = {
        'Recall': class_report['Tumour']['recall'],
        'Precision': class_report['Tumour']['precision'],
        'F1_score': class_report['Tumour']['f1-score'],
        'AUC_score': roc_auc_score(y_te, y_prob),
        'Optimal threshold': thresh,
        }
    
    predicts[new_name] = {
        'y_pred': y_pred_opt,
        'y_prob': y_prob,
        'report_df': pd.DataFrame(class_report).T
    }

df = pd.DataFrame(scores).T

st.markdown(make_markdown_table(df))

st.markdown("---")

st.markdown("##### Performance Curves")

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    
    for name, model in scores.items():
        RocCurveDisplay.from_predictions(
            y_te, predicts[name]['y_prob'],
            name=f"{name} (AUC={scores[name]['AUC_score']:.3f})",
            ax=ax_roc
        )

    ax_roc.plot([0,1], [0,1], 'k--', label='Random baseline')
    ax_roc.set_title('ROC Curves — All Models')
    ax_roc.legend(loc='lower right')
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlabel("False Positive Rate")
    st.pyplot(fig_roc)
    plt.close(fig_roc)

with comp_col2:
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for name, res in predicts.items():

        PrecisionRecallDisplay.from_predictions(
            y_te,
            res['y_prob'],
            name=name,
            ax=ax_pr
        )
    ax_pr.set_title('Precision-Recall — All Models')
    st.pyplot(fig_pr)
    plt.close(fig_pr)

st.markdown("---")


model_choice = st.selectbox("Select model to inspect:", 
                            ["Logistic Regression", "XGBoost", "Random Forest"]
)

tab1, tab2, tab3, tab4 = st.tabs(['Evaluation', 'Threshold', 'Features', 'CV Results'])


# Evaluation
with tab1:
    st.markdown("##### Evaluation charts")

    eval_col1, eval_col2 = st.columns(2)
    
    with eval_col1:
        cm_fig, cm_ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(y_te, predicts[model_choice]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=cm_ax,
            xticklabels=['No Tumour', 'Tumour'],
            yticklabels=['No Tumour', 'Tumour'])
        cm_ax.set_title(f'Confusion Matrix')
        cm_ax.set_ylabel('Actual')
        cm_ax.set_xlabel('Predicted')
        st.pyplot(cm_fig)
        plt.close(cm_fig)

    with eval_col2:
        prob_dist_fig, prob_dist_ax = plt.subplots(figsize=(8, 6.5))
        thresh_val = thresholds[inv_names[model_choice]]
        # 1. Plot overlapping histograms split by class
        pred_df = pd.DataFrame({
            'y_true': y_te,
            'y_prob': predicts[model_choice]['y_prob']

        })
        sns.histplot(
            data=pred_df,
            x='y_prob',
            hue='y_true',
            bins=20,
            element='step',      # Draws crisp outlines instead of heavy overlapping bars
            stat='count',   
            common_norm=False,   # Compares shapes fairly even with class imbalance
            ax=prob_dist_ax,
        )

        prob_dist_ax.axvline(x=thresh_val, color='red', linestyle='--', label=f'Threshold ({thresh_val:.2f})')
        custom_lines = [
            Line2D([0], [0], color='C0', lw=2, label='Healthy (No Tumor)'),     # C0 = default blue
            Line2D([0], [0], color='C1', lw=2, label='Tumor Detected'),        # C1 = default orange
            Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'Threshold ({thresh_val:.2f})')
        ]

        prob_dist_ax.set_title("Probability Distribution by Class")
        prob_dist_ax.set_xlabel("Predicted Probability of Tumor")
        prob_dist_ax.set_ylabel("Density")
        prob_dist_ax.legend(handles=custom_lines, loc='upper right')
        st.pyplot(prob_dist_fig)
        plt.close(prob_dist_fig) 

    st.markdown("---")

    st.markdown("##### Classification Report")

    st.dataframe(predicts[model_choice]['report_df'].style.format("{:.3f}")) 

# Threshold
with tab2:

    st.markdown("##### Scores vs threshold chart")

    precisions_opt, recalls_opt, thresholds_opt = precision_recall_curve(
        y_te, 
        predicts[model_choice]['y_prob'])
    # removing the last element so it matches the size of the thresholds_opt (there is N-1 thersholds)    
    precisions_opt = precisions_opt[:-1]
    recalls_opt = recalls_opt[:-1]
    f1_opt = np.divide(
        2 * (precisions_opt * recalls_opt),
        precisions_opt + recalls_opt,
        out=np.zeros_like(precisions_opt),
        where=(precisions_opt + recalls_opt) != 0
    )
    df_thresh = pd.DataFrame({
        'Threshold': thresholds_opt,
        'Precision': precisions_opt,
        'Recall': recalls_opt,
        'F1_score': f1_opt
    })

    fig_comp_thresh, axes_comp_thresh = plt.subplots(figsize=(14, 5))

    axes_comp_thresh.plot(thresholds_opt, recalls_opt,    label='Recall',    color='tomato',    lw=2)
    axes_comp_thresh.plot(thresholds_opt, precisions_opt, label='Precision', color='steelblue', lw=2)
    axes_comp_thresh.plot(thresholds_opt, f1_opt,  label='F1',        color='green',     lw=2)
    axes_comp_thresh.axvline(0.5,          color='gray',   linestyle='--', label='Default (0.5)')
    axes_comp_thresh.axvline(thresh_val,  color='black',  linestyle='--', label=f'Optimal ({thresh_val:.2f})')
    axes_comp_thresh.set_xlabel('Threshold')
    axes_comp_thresh.set_ylabel('Score')
    axes_comp_thresh.set_title('Metrics vs Threshold')
    axes_comp_thresh.legend()
    axes_comp_thresh.grid(alpha=0.3)

    st.pyplot(fig_comp_thresh)
    plt.close(fig_comp_thresh)

    # Highlight key threshold stats
    st.metric("Selected Threshold", f"{thresh_val:.2f}")
  
    st.markdown("<br>", unsafe_allow_html=True)

    # Explanation Box
    st.info(
        "**Why this threshold?**\n\n"
        "Our optimal threshold is selected by finding the point that **maximizes Recall** "
        "while preserving the highest possible Precision. This guarantees that all potential tumor cases "
        "are flagged for clinical review without generating excessive unnecessary alarms."
    )

    st.markdown("---")

    st.markdown("##### Table of threshold values with scores")

    def highlight_optimal(row):
        if row['Threshold'] == thresh_val:
            return ['background-color: lightgreen; color: black'] * len(row)
        return [''] * len(row)

    styled_df = df_thresh.style.apply(highlight_optimal, axis=1)

    st.dataframe(styled_df)


# Features
with tab3:
    st.markdown("##### Features importance analysis")

    model_key = inv_names[model_choice]
    model = models[inv_names[model_choice]]
    importance_df = get_features_importance(model, feature_names)
    fig_feat, ax_feat = plt.subplots(figsize=(10, 6))

    if model_choice == "Logistic Regression":
        colors = ['steelblue' if s > 0 else 'tomato' for s in importance_df['Importance']]
        xlabel = 'Log-Odds Coefficient (+ Risk / - Protective)'
        ax_feat.axvline(0, color='black', linestyle='--', linewidth=0.8)
    else:
        colors = 'steelblue'
        xlabel = 'Gain' if model_key == 'xgb' else 'Log loss'
        
    importance_df.plot(kind='barh', x='Feature', y='Importance',
                        ax=ax_feat, color=colors, legend=False)
    ax_feat.set_title(f'{model_choice} Feature Importance')
    ax_feat.set_xlabel(xlabel)

    ax_feat.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_feat)
    plt.close(fig_feat)

# CV results
with tab4:
    st.markdown("##### ROC per fold chart")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = models[inv_names[model_choice]]

    # generate and display ROC fold plot dynamically per selected model
    roc_fold_fig, fold_df, fold_summary = generate_roc_cv_fold_plot(
        estimator=model,
        cv=cv,
        X=X_tr,
        y=y_tr,
        model_name=model_choice
    )

    st.pyplot(roc_fold_fig)
    plt.close(roc_fold_fig)
    
    st.markdown("---")


    st.markdown("##### Mean ± Std Summary")

    fold_col1, fold_col2 = st.columns(2)
    fold_col1.metric("Mean AUC", f"{fold_summary['mean_auc']:.3f} ± {fold_summary['std_auc']:.3f}")
    fold_col2.metric("Mean Recall", f"{fold_summary['mean_recall']:.3f} ± {fold_summary['std_recall']:.3f}")  

    st.markdown("---")

    st.markdown("##### Fold Scores Table")

    st.dataframe(
        fold_df.style.format({"AUC": "{:.3f}", "Recall": "{:.3f}"}),
        hide_index=True
    )


