import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from depth2cloud import *
from PIL import Image

def save_uploaded_file(uploaded_file, save_path='static/images'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        save_file_path = os.path.join(save_path,uploaded_file.name)
        with open(os.path.join(save_path,uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return save_file_path   
    except:
        return None

save_path = 'static/images'

depth_save_dir = "depth_output"
segmentation_model_path = "./segmentation_model/pointrend_resnet50.pkl"
depth_model_base_path = './midas_depth/weights/'
cam_mat_save_path = os.path.join('cam_matrix/cameraIntrinsic_apple.xml')
cloud_save_dir = "./point_clouds"

st.title('DDP-1: 2D to 3D converter')
valid_molecule = True
loaded_molecule = None
selection = None
submit = None

# page = st.sidebar.selectbox('Page Navigation', ["3D model estimation", "Under the hood"])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [RA Keerthan](https://github.com/keerthan2)")
st.sidebar.image("static/logo.png", width=100)

st.markdown("Select input RGB Image")
upload_columns = st.columns([2, 1])
file_upload = upload_columns[0].expander(label="Upload a RGB image")
uploaded_file = file_upload.file_uploader("Choose a RGB image", type=['jpg','png','jpeg'])
# uploaded_file = st.file_uploader("Upload a RGB Image")

if uploaded_file is not None:
    save_file_path = save_uploaded_file(uploaded_file)
    if save_file_path is not None: 
        display_image = Image.open(uploaded_file)
        st.info("This image appears to be valid :ballot_box_with_check:")
        upload_columns[1].image(display_image)
        submit = upload_columns[1].button("Run 3D model generation")
        st.markdown("""---""")
        if submit:
            with st.spinner(text="Fetching the 3D model..."):
                img_path = save_file_path
                depth_calibration_pipeline = predict(img_path, depth_save_dir,
                                                    segmentation_model_path = segmentation_model_path, 
                                                    depth_model_base_path = depth_model_base_path,
                                                    cam_mat_save_path = cam_mat_save_path,
                                                    cloud_save_dir = cloud_save_dir)
            pcd_file_name = f"{uploaded_file.name.split('.')[0]}.ply"
            pcd_path = os.path.join(cloud_save_dir, pcd_file_name)
            with open(pcd_path, 'rb') as f:
                st.download_button(label = 'Download 3D Model', data = f, file_name=pcd_file_name)
    else:
        st.error("This image appears to be invalid :no_entry_sign:")