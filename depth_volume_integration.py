from imports import *
from integration_utils import *

segmentation_model_path = "./segmentation_model/pointrend_resnet50.pkl"
depth_model_base_path = './midas_depth/weights/'

img_path = "imgs/pixel2world/b0p729.jpg"
fname = img_path.split('/')[-1]
depth_save_dir = "depth_output"
depth_path = f'./{depth_save_dir}/{fname}'
depth_pfm_path = f"./{depth_save_dir}/{fname.split('.')[0]}"+'.pfm'
cam_mat_path = 'cam_matrix/cameraIntrinsic_apple.xml'
cloud_save_dir = "./point_clouds"

# settings for scaling
gt_background_depth = 66.9 #cm
margin = 50

voxel_size = 7e-7

depth_calibration_pipeline = DepthCalibrationPipeline(segmentation_model_path, depth_model_base_path)

depth_calibration_pipeline.run_base_block(img_path, depth_save_dir)

centered = 1
transformed_cloud_o3d = obj2cloud(depth_calibration_pipeline, gt_background_depth=gt_background_depth, 
                  intrinsics_mat_path=cam_mat_path, margin=margin, centered=centered,
                  voxel_size=voxel_size, cloud_save_dir=None)

cloud_save_dir = "./point_clouds"
if not os.path.exists(cloud_save_dir):
  os.makedirs(cloud_save_dir)
filename = img_path.split('/')[-1].split('.')[0]+'.ply'
o3d.io.write_point_cloud(os.path.join(cloud_save_dir,filename), transformed_cloud_o3d, 
                            write_ascii=False, compressed=False, print_progress=False)