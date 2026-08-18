import streamlit as st

if not st.session_state:
    st.session_state.role = 'clinical'

with st.sidebar:
    # We use HTML inside st.markdown to get absolute control over font size and styling
    mode_title = f"> {st.session_state.role.upper()} MODE <"
    st.markdown(
        f'<p style="font-size:20px; font-weight:bold; margin-bottom:0px; color:yellow">{mode_title}</p>', 
        unsafe_allow_html=True
    )
    
    # Simple separator line
    st.markdown("---")
    
    # The Switch Button sits nicely underneath the big title
    if st.button('Switch Mode', use_container_width=True):
        st.session_state.role = 'research' if st.session_state.role == 'clinical' else 'clinical'
        st.rerun()

# define pages
home_page    = st.Page('pages/home.py',    title='Home',     icon='🏠', default=True)
predict_page = st.Page('pages/predict.py', title='Predict')      
eda_page     = st.Page('pages/eda.py',     title='EDA')
models_page  = st.Page('pages/models.py', title='Model Comparison')
shap_page    = st.Page('pages/shap_page.py', title='SHAP Analysis')


# build navigation based on role
if st.session_state.role == 'clinical':
    pg = st.navigation([home_page, predict_page])

else:  # research
    pg = st.navigation([home_page, eda_page, models_page, shap_page])
    

if pg.title != 'Home':
    st.title(f"{st.session_state.role.capitalize()} Mode")
    st.markdown("---")


pg.run()