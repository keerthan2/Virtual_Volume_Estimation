import streamlit as st
import pandas as pd
from utils_dashboard import *

form = st.form(key='testForm')

something = form.text_input("Input Something",value="", key="FORM_INPUT")

def handle_form_submit():
    # st.session_state.submission = st.session_state.FORM_INPUT
    st.session_state.submission = True

submit_button = form.form_submit_button(label='Submit', on_click=handle_form_submit)

if 'submission' in st.session_state:
    st.markdown("This text is printed thanks to you having submitted the form.")
    st.markdown(f"Here is what you wrote: {something}")

contentDF = ["One","Two","Three"]
dataframeFinal = pd.DataFrame(contentDF)
csv = dataframeFinal.to_csv(index=True)

st.session_state['download'] = st.download_button(
    label="Download CSV",
    data=csv,
    mime="text/csv",
    file_name="CSV.csv")

output = st.empty()
with st_capture(output.code):
    # print(f"{st.session_state['save_file_path']}")
    print(st.session_state['submission'], st.session_state['download'])