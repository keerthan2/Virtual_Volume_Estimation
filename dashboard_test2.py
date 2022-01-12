import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from depth2cloud import *
from PIL import Image
from utils_dashboard import *
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
save_path = 'static/images'

depth_save_dir = "depth_output"
segmentation_model_path = "./segmentation_model/pointrend_resnet50.pkl"
depth_model_base_path = './midas_depth/weights/'
cam_mat_save_path = os.path.join('cam_matrix/cameraIntrinsic_apple.xml')
cloud_save_dir = "./point_clouds"

st.title('DDP-1: 2D to 3D converter')

# st.session_state.upl_names = []

def handle_submit():
    st.session_state.submit = True

def handle_upload():
    # if 'submit' in st.session_state:
    #     del st.session_state.submit

    # st.session_state.submit = False
    if 'upl_names' not in st.session_state:
        st.session_state.upl_names = [st.session_state.upl]
    else:
        st.session_state.upl_names.append(st.session_state.upl)

# page = st.sidebar.selectbox('Page Navigation', ["3D model estimation", "Under the hood"])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [RA Keerthan](https://github.com/keerthan2)")
st.sidebar.image("static/logo.png", width=100)

st.markdown("Select input RGB Image")
upload_columns = st.columns([2, 1])
file_upload = upload_columns[0].expander(label="Upload a RGB image")
uploaded_file = file_upload.file_uploader("Choose a RGB image", type=['jpg','png','jpeg'], key='upl', on_change=handle_upload)

if uploaded_file is not None:
    save_file_path = save_uploaded_file(uploaded_file)
    if save_file_path is not None: 
        st.session_state['save_file_path'] = save_file_path
        st.session_state['upload_file_name'] = uploaded_file.name
        display_image = Image.open(uploaded_file)
        st.info("This image appears to be valid :ballot_box_with_check:")
        upload_columns[1].image(display_image)
        submit_button = upload_columns[1].button("Submit", on_click = handle_submit)
        st.markdown("""---""")
    else:
        st.error("This image appears to be invalid :no_entry_sign:")

if ('upl_names' in st.session_state) and (len(st.session_state.upl_names) > 1):
    try:
        if st.session_state.upl_names[-1].id == st.session_state.upl_names[-2].id:
            with open(save_file_path, 'rb') as f:
                st.download_button(label = 'Download Image', data = f, file_name=uploaded_file.name)
        else:
            st.session_state.upl_names = [st.session_state.upl_names[-1]]
    except:
        st.session_state.upl_names = [st.session_state.upl_names[-1]]
else:
    if 'submit' in  st.session_state:
        with open(save_file_path, 'rb') as f:
            st.download_button(label = 'Download Image', data = f, file_name=uploaded_file.name)




