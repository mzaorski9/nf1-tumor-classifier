import streamlit as st

st.markdown("""
### What is NF1?

Neurofibromatosis Type 1 (NF1) is a genetic disorder affecting about 
1 in 3,000 people. It causes tumour growth along nerve tissue and 
a range of clinical symptoms — café-au-lait skin spots, freckling, 
neurofibromas, and in some cases optic gliomas or other tumours.

Roughly half of NF1 cases are inherited (**familial**), and half 
arise from spontaneous mutation (**sporadic**).
""")

st.markdown("---")

st.markdown("""
### What does this app do?

This tool predicts the likelihood that an NF1 patient will develop 
a tumour, based on their clinical symptom profile — using a 
model trained on real patient data.

It also lets you explore:
- How the underlying dataset is distributed
- How three different Machine Learning models compare on this task
- **Why** the model makes each individual prediction
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Patients in dataset", "296")
with col2:
    st.metric("Clinical features", "18")
with col3:
    st.metric("Model recall (tumour)", "100%")

st.markdown("---")

st.warning("""
**Disclaimer:** This tool is built for educational and portfolio 
purposes only. It is **not** a diagnostic tool and must not be used 
for real clinical decision-making. Always consult a qualified 
medical professional.
""")

st.caption("""
Dataset: Sharafi, P., Arslan, H., Ersoy, S., Varan, A., & Ayter, Ş. (2025). 
Neurofibromatosis Type 1; Clinical Symptoms of Familial and Sporadic Cases. 
UCI Machine Learning Repository.
""")