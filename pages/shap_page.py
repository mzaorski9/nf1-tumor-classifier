
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import (
    load_data, 
    load_models, 
    load_thresholds, 
    load_y_probs,
    load_preprocessor
)
from src.explainability import (
    get_shap_values,
    plot_beeswarm, 
    plot_bar_chart,
    plot_waterfall,
    get_shap_ranking
)

models     = load_models()      # dict
y_probs    = load_y_probs()
thresholds = load_thresholds()  # dict
X_te, y_te, X_tr, y_tr = load_data()
preprocessor = load_preprocessor()

mapped_names = {'lr': 'Logistic Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
inv_names = { v : k for k, v in mapped_names.items()}

feature_names = [
    'Case_Type', 'Age_of_Mother', 'Age_of_Father', 'Age_at_First_Diagnosis',
    'Café_au_lait_CLS', 'Axillary_Freckles', 'Inguinal_Freckles', 'Lisch_Nodules',
    'Dermal_Neurofibromins', 'Plexiform_Neurofibromins', 'Optic_Glioma',
    'Skeletal_Dysplasia', 'Learning_Disability', 'Hypertension', 'Astrocytoma',
    'Hamartoma', 'Scoliosis', 'Other_Symptoms'
]


st.subheader("Model Explainability — SHAP")


st.markdown("<br>", unsafe_allow_html=True)


tab1, tab2 = st.tabs(['Global', 'Individual/Comperative'])

# Global
with tab1:

    model_choice_glob = st.selectbox("Select model to inspect:", 
                                ["Logistic Regression", "XGBoost", "Random Forest"], 
                                key="model_choice_tab1"
    )
    st.markdown("##### Feature Impact Overview")

    model_key_glob = inv_names[model_choice_glob]
    model_glob = models[model_key_glob]['model']

    if model_choice_glob == "Logistic Regression":
        shap_vals = get_shap_values(model_glob, X_te, "Linear", preprocessor)

    elif model_choice_glob == "Random Forest" or model_choice_glob == "XGBoost":
        shap_vals = get_shap_values(model_glob, X_te, "Tree")

    else:
        raise ValueError("No input data provided or invalid model type: choose 'Linear' or 'Tree'.")
    
    bees_col, bar_col = st.columns(2)


    with bees_col:
        bees_fig = plot_beeswarm(shap_vals, model_choice_glob)
        st.pyplot(bees_fig)
        plt.close(bees_fig)

    with bar_col:
        bar_fig = plot_bar_chart(shap_vals, model_choice_glob)
        st.pyplot(bar_fig)
        plt.close(bar_fig)

        bar_dataframe = get_shap_ranking(shap_vals)

        st.markdown("##### Summary SHAP Table")

        # display Mean |SHAP| column as .5f
        st.dataframe(bar_dataframe.style.format({"Mean |SHAP|":"{:.5f}"}))


# Individual/Comparative
with tab2:
    compare = st.toggle("Comparative Mode")

    st.markdown("<br>", unsafe_allow_html=True)

    model_choice_indiv = st.selectbox("Select model to inspect:", 
                            ["Logistic Regression", "XGBoost", "Random Forest"], 
                            key="model_choice_tab2"

    )
    model_key_indiv = inv_names[model_choice_indiv]
    model_indiv = models[model_key_indiv]['model']

    patients_df = pd.DataFrame({
        "True_label": y_te,
        "Predicted_prob_LR": y_probs['lr'],
        "Predicted_prob_RF": y_probs['rf'],
        "Predicted_prob_XGB": y_probs['xgb']
    }).sort_index()

    # add the necessary columns (for Comparative Mode)
    patients_df["Pred_LR"] = (patients_df['Predicted_prob_LR'] >= thresholds['lr']).astype(int)
    patients_df["Pred_RF"] = (patients_df['Predicted_prob_RF'] >= thresholds['rf']).astype(int)
    patients_df["Pred_XGB"] = (patients_df['Predicted_prob_XGB'] >= thresholds['xgb']).astype(int)

    if compare:
        format_func = lambda idx: (
            f"Patient #{idx}  |  True: {patients_df.loc[idx, 'True_label']}  |  "
            f"LR: {patients_df.loc[idx, 'Predicted_prob_LR']:.2%}  |  "
            f"RF: {patients_df.loc[idx, 'Predicted_prob_RF']:.2%}  |  "
            f"XGB: {patients_df.loc[idx, 'Predicted_prob_XGB']:.2%}"
        )
    else:
        format_func = lambda idx: (
            f"Patient #{idx}  |  True: {patients_df.loc[idx, 'True_label']}  |  "
            f"Prob: {patients_df.loc[idx, f'Predicted_prob_{model_key_indiv.upper()}']:.2%}"
        )    

    patient_idx = st.selectbox(
        "Select a patient:",
        options=patients_df.index,
        format_func=format_func
    )

    st.markdown("---")

    selected_patient_data = X_te.loc[[patient_idx]] # as DataFrame row
    formatted_patient_data = selected_patient_data.T.rename(columns={selected_patient_data.index[0]: "Value"})

    models_shap_vals = {}

    for key, pipeline in models.items():
        if key == 'lr':
            models_shap_vals[key] = get_shap_values(pipeline['model'], selected_patient_data, "Linear", preprocessor, X_te)
            # re-writting unscaled Ages data (Logistic Regression)
            models_shap_vals[key].data = selected_patient_data.values
        elif key == 'rf' or key == 'xgb':
            models_shap_vals[key] = get_shap_values(pipeline['model'], selected_patient_data, "Tree")  
        else:
            raise ValueError("No input data provided or invalid model type: choose 'Linear' or 'Tree'.")
        
    if compare:
        st.markdown("##### Comperative Patient Analysis")

        # plot all waterfall charts in n_cols 
        models_keys = list(models.keys())
        n_cols = 3

        for i in range(0, len(models_keys), n_cols):
            selected = models_keys[i: i + n_cols]
            cols = st.columns(n_cols)
            for key, col in zip(selected, cols):
                with col:
                    fig = plot_waterfall(shap_values=models_shap_vals[key], 
                                        title=f"Waterfall Plot ({mapped_names[key]})", 
                                        clinical_mode=False) 
                    st.pyplot(fig)
                    plt.close(fig)

        filter_option = st.radio(
            "Filter patients:",
            ["All", "Any model misclassified", "Models disagree with each other"]
        )

        if filter_option == "Any model misclassified":
            filtered_rows = patients_df[['Pred_LR', 'Pred_RF', 'Pred_XGB']].ne(patients_df['True_label'], axis=0)
            patients_df = patients_df[filtered_rows.any(axis=1)]
            
        elif filter_option == "Models disagree with each other":
            filtered_rows = patients_df[['Pred_LR', 'Pred_RF', 'Pred_XGB']].nunique(axis=1) > 1
            patients_df = patients_df[filtered_rows]

        st.dataframe(patients_df)
    else:
        st.markdown("##### Individual Patient Analysis")
        
        watefall_fig = plot_waterfall(shap_values=models_shap_vals[model_key_indiv], 
                            title=f"Waterfall Plot ({model_choice_indiv})", 
                            clinical_mode=False) 
        st.pyplot(watefall_fig)
        plt.close(watefall_fig) 

        st.markdown("##### Patient's details:")

        st.table(
            data=formatted_patient_data,
            width='content',
        )

    
