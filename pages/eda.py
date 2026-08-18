import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import sys
sys.path.append('src')
from  src.data_loader import load_data


def get_data():
    X_te, y_te, X_tr, y_tr = load_data()
    X = pd.concat([X_tr, X_te]).sort_index()
    y = pd.concat([y_tr, y_te]).sort_index()
    return X, y

X, y = get_data()

symptom_cols = [
    'Café_au_lait_CLS', 'Axillary_Freckles', 'Inguinal_Freckles',
    'Lisch_Nodules', 'Dermal_Neurofibromins', 'Plexiform_Neurofibromins',
    'Optic_Glioma', 'Skeletal_Dysplasia', 'Learning_Disability',
    'Hypertension', 'Astrocytoma', 'Hamartoma', 'Scoliosis', 'Other_Symptoms'
]
age_cols = ['Age_of_Mother', 'Age_of_Father', 'Age_at_First_Diagnosis']


st.subheader('Exploratory Data Analysis')

tab1, tab2, tab3 = st.tabs(['Overview', 'Features', 'Patient Explorer'])

# Overview
with tab1:

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients",   len(X))
    col2.metric("Features",         X.shape[1])
    col3.metric("No Tumour",        int((y == 0).sum()))
    col4.metric("Tumour",           int((y == 1).sum()))

    st.markdown("---")
    
    st.markdown("##### Class Distribution")

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    ax.bar(['No Tumour', 'Tumour'], counts.values,
            width=0.5, color=['steelblue', 'red'], edgecolor='black')
   
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title('Target Distribution')
    st.pyplot(fig)
    plt.close(fig)

    
    st.markdown("##### Raw Data Sample")
    df_display = pd.concat([X, y], axis=1)
    st.dataframe(df_display, height=350)

    st.markdown("---")

    ratio = (y == 0).sum() / (y == 1).sum()
    st.info(f"**Imbalance ratio:** {ratio:.2f}x more No-Tumour than Tumour cases. "
            f"Handled via `class_weight='balanced'` during training.")

# Features
with tab2:
    sym_means = X[symptom_cols].mean().sort_values(ascending=False)

    st.markdown("##### Symptom prevalence")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sym_means.index, sym_means.values, width=0.5)
    ax.set_xlabel("Symptoms")
    ax.set_ylabel("Proportion of patients")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.markdown("##### Symptom Rates: Tumour vs No Tumour")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(symptom_cols))
    width = 0.35
    temp_X = X.copy()
    temp_X['target'] = y
    means = temp_X.groupby('target')[symptom_cols].mean()

    bars1 = ax.bar(x - width/2, means.loc[0], width, label='No Tumour', 
                color='steelblue')
    bars2 = ax.bar(x + width/2, means.loc[1], width, label='Tumour', 
                color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels(symptom_cols, rotation=45, ha='right')
    ax.set_title('Symptom Rates: Tumour vs No Tumour')
    ax.set_ylabel('Proportion of Patients')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.markdown("##### Symptom Correlation Matrix")

    fig, ax = plt.subplots(figsize=(12, 8))
    corr = X[symptom_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, square=True)
    ax.set_title('Symptom Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.markdown("##### Age Distributions by Class")

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    for ax, col in zip(axes, age_cols):
        bin = np.linspace(X[col].min(), X[col].max(), 20)
        ax.hist(X[col][y == 0], bins=bin, alpha=0.6, color='steelblue', label='No Tumour')
        ax.hist(X[col][y == 1], bins=bin, alpha=0.6, color='tomato', label='Tumour')
        ax.set_title(col)
        ax.set_ylabel('Count')
        ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Chi-square / Welch's Test
    st.markdown("##### Statistical Significance Table")

    sig_temp_results = []

    for col in symptom_cols:
        crosstab = pd.crosstab(X[col], y)
        stat, pval, dot, expect_freq = chi2_contingency(crosstab)
        chi_data = {
            'Feature name': col.replace('_', ' '),
            'Score': round(stat, 3),
            'P-value': round(pval, 3),
            'Is significant (p<0.05)': 'yes' if pval < 0.05 else 'no'
        }
        sig_temp_results.append(chi_data)
    
    for col in age_cols:
        pos_group = X[y == 1][col]
        neg_group = X[y == 0][col]
        score, pval2 = ttest_ind(
                        pos_group,
                        neg_group, 
                        equal_var=False,    # Welch's test
                        nan_policy='omit'
        )
        welch_data = {
            'Feature name': col.replace('_', ' '),
            'Score': round(score, 3),
            'P-value': round(pval2, 3),
            'Is significant (p<0.05)': 'yes' if pval2 < 0.05 else 'no'
        }
        sig_temp_results.append(welch_data)

    sig_df = pd.DataFrame(sig_temp_results)
    sig_df = sig_df.fillna('---')

    st.dataframe(sig_df, use_container_width=True)


# Patients explorer
with tab3:

    st.markdown("##### Filter patients by symptom profile")

    col_f1, col_f2 = st.columns([2, 1])

    full_data = pd.concat([X, y], axis=1)
    filtered = full_data.copy()

    with col_f1:
        selected_symptoms = st.multiselect(
            'Symptoms that must be PRESENT (1):',
            options=symptom_cols,
            format_func=lambda x: x.replace('_', ' ')
        )

        selected_ages = st.multiselect(
            'Filter by age:',
            options=age_cols,
            format_func=lambda x: x.replace('_', ' ')
        )

        for age_inpt in selected_ages:
            if age_inpt == "Age_at_First_Diagnosis":
                age_dx = st.slider("Age at First Diagnosis", 0, 100, 5)
            else:
                age_dx = st.slider(f"{age_inpt.replace('_', ' ')}", 15, 70, 30)
            
            filtered = filtered[filtered[age_inpt] == age_dx]

    with col_f2:
        class_filter = st.radio(
            'Class filter:',
            ['All', 'Tumour only', 'No Tumour only']
        )

    for col in selected_symptoms:
        filtered = filtered[filtered[col] == 1]

    if class_filter == 'Tumour only':
        filtered = filtered[filtered['Tumour_Case'] == 1]
    elif class_filter == 'No Tumour only':
        filtered = filtered[filtered['Tumour_Case'] == 0]
    

    st.markdown("---")

    # summary metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("Matches: ", len(filtered))

    if len(filtered) > 0:
        tumour_cases = filtered['Tumour_Case']
        c2.metric("Tumour cases in group ", int(tumour_cases.sum()))
        c3.metric("Tumour cases in group [%] ", f"{tumour_cases.mean():.0%}")
    else:
        c2.metric("Tumour cases in group ", "N/A")
        c3.metric("Tumour cases in group [%] ", "N/A")

    formatted_df = filtered.copy()

    formatted_df = formatted_df.rename(columns=lambda x: x.replace('_', ' '))

    st.dataframe(formatted_df, height=600)


