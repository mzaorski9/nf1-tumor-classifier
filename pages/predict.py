import streamlit as st
import pandas as pd
from src.data_loader import load_models, load_thresholds, load_scaler,  load_data
from src.model import predict_with_threshold, get_risk_level, generate_result_comment, get_pred_contributors
from src.explainability import get_shap_values, plot_waterfall



models = load_models()
thresholds = load_thresholds()
scaler = load_scaler()
data = load_data()


if 'role' not in st.session_state: 

    st.session_state.role = 'clinical'

# CLINICAL VIEW 

if st.session_state.role == 'clinical':

    with st.form("nf1_prediction_form"):

        st.markdown("#### Patient Demographics & History")
        
        demo_col1, demo_col2, demo_col3  = st.columns([1, 0.5, 1.2])
        
        with demo_col1:
            age_dx = st.slider("Age at Diagnosis", 0, 100, 5)
            age_mother = st.slider("Mother's Age at Birth", 15, 60, 28)
            age_father = st.slider("Father's Age at Birth", 15, 70, 32)
        with demo_col3:
            # Putting Case Type right next to demographics
            case_type_text = st.radio(
                "Case Type (Family History)", 
                ["Sporadic (No Family History)", "Familial (Parent has NF1)"]
            )
            case_type = 1 if "Familial" in case_type_text else 0

        st.markdown("---")

        st.markdown("#### Clinical Observations (Present Symptoms)")
        st.write("Check all hallmark signs observed during the physical and ocular examination:")
        
        sym_col1, sym_col2, sym_col3 = st.columns(3)
        
        with sym_col1:
            cafe = st.checkbox("Café au lait spots (CLS)")
            axillary = st.checkbox("Axillary Freckles")
            inguinal = st.checkbox("Inguinal Freckles")
            lisch = st.checkbox("Lisch Nodules")
            dermal = st.checkbox("Dermal Neurofibromas")

        with sym_col2:
            plexiform = st.checkbox("Plexiform Neurofibromas")
            optic = st.checkbox("Optic Glioma")
            skeletal = st.checkbox("Skeletal Dysplasia")
            scoliosis = st.checkbox("Scoliosis")
            learning = st.checkbox("Learning Disabilities")

        with sym_col3:
            astrocytoma = st.checkbox("Astrocytoma")
            hamartoma   = st.checkbox("Hamartoma")
            htn         = st.checkbox("Hypertension")
            other_sym   = st.checkbox("Other Related Symptoms")

        st.markdown("---")

        submit_btn = st.form_submit_button("Generate Risk Assessment", use_container_width=True)
        
        if submit_btn:
            # build the feature row in EXACT column order the model expects
            patient_data = pd.DataFrame([{
                'Case_Type':                 case_type,
                'Age_of_Mother':              age_mother,
                'Age_of_Father':              age_father,
                'Age_at_First_Diagnosis':     age_dx,
                'Café_au_lait_CLS':           int(cafe),
                'Axillary_Freckles':          int(axillary),
                'Inguinal_Freckles':          int(inguinal),
                'Lisch_Nodules':              int(lisch),
                'Dermal_Neurofibromins':      int(dermal),
                'Plexiform_Neurofibromins':   int(plexiform),
                'Optic_Glioma':               int(optic),
                'Skeletal_Dysplasia':         int(skeletal),
                'Learning_Disability':        int(learning),
                'Hypertension':               int(htn),
                'Astrocytoma':                int(astrocytoma),
                'Hamartoma':                  int(hamartoma),
                'Scoliosis':                  int(scoliosis),
                'Other_Symptoms':             int(other_sym),
            }])

            try:
                # models are pipelines, as we dumped them this way
                pipeline = models['lr']
                threshold = thresholds['lr']

                y_pred, y_prob = predict_with_threshold(patient_data, pipeline, threshold)
                pred_warn_lvl = get_risk_level(y_prob, threshold)

            except (RuntimeError, ValueError, TypeError) as e:
                st.error(f"Validation error occured: {e}")    
            except Exception as e:
                st.error(f"Unexpected error occured: {e}")

            st.divider()
            col1, col2, col3 = st.columns([2, 1, 1])

            # get (single) element from the 'y_prob' array and convert to Python scalar
            risk = y_prob.item() if hasattr(y_pred, 'item') else float(y_prob[0])
            
            with col1:
                st.markdown(f"<h3 style='color: orange;'>{pred_warn_lvl} </h3>", unsafe_allow_html=True)
            with col2:
                st.markdown("<p style='color: orange; font-weight: 600; margin-bottom: 0;'>Predicted chance of tumor: </p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: orange; font-size: 2rem; margin: 0;'>{risk*100:.1f}%</p>", unsafe_allow_html=True)
            
            st.divider()

            model = pipeline.named_steps['model']
            background_data = data[2]       # X_train
  
            shap_vals = get_shap_values(model, scaler, patient_data, "Linear", background_data)
            fig = plot_waterfall(shap_values=shap_vals, title="Symptoms relevance", clinical_mode=True)

            st.pyplot(fig)
            st.divider()

            risk_factors, protect_factors = get_pred_contributors(shap_vals, patient_data, 3)
            comment = generate_result_comment(risk_factors, protect_factors, pred_warn_lvl, patient_data)

            st.markdown(comment)




