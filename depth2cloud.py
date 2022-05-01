from imports import *
from integration_utils import *
from depth_estimation import *
from pc_functions import *

# img_path = "imgs/sand_example.jpeg"


def predict(img_path, depth_save_dir,
    segmentation_model_path = "./segmentation_model/pointrend_resnet50.pkl", 
    depth_model_base_path = './midas_depth/weights/',
    cam_mat_save_path = os.path.join('cam_matrix/cameraIntrinsic_apple.xml'),
    cloud_save_dir = "./point_clouds",
    gt_background_depth = 66.9,
    margin = 50,
    voxel_size = 7e-7,
    centered = 1,
    isCalib = False,
    isVoxelDown = False,
    isMeshGen = False, 
    isVolEst = False
    ):
    if isCalib:
        calib_img_dir = os.path.join('./calib_imgs_apple')
        results_save_dir = os.path.join('./calib_imgs_results')
        if not os.path.exists(results_save_dir):
            os.makedirs(results_save_dir)
        chessboardSize = (24,17)
        calibrator(calib_img_dir, results_save_dir, cam_mat_save_path=cam_mat_save_path, chessboardSize = chessboardSize)


    depth_calibration_pipeline = DepthCalibrationPipeline(segmentation_model_path, depth_model_base_path)
    depth_calibration_pipeline.run_base_block(img_path, depth_save_dir)
    transformed_cloud_o3d = obj2cloud(depth_calibration_pipeline, gt_background_depth=gt_background_depth, 
                    intrinsics_mat_path=cam_mat_save_path, margin=margin, centered=centered,
                    cloud_save_dir=None)

    # depth_calibration_pipeline = DepthCalibrationPipeline(segmentation_model_path, depth_model_base_path)
    # depth_calibration_pipeline.run_base_block(img_path, depth_save_dir)

    # cloud = obj2cloud(depth_calibration_pipeline, "cake", intrinsics_mat_path=cam_mat_save_path)

    # transformed_cloud = cloud
    # transformed_cloud_np = transformed_cloud.points.to_numpy()
    # transformed_cloud_np = transformed_cloud_np[~np.isnan(transformed_cloud_np).any(axis=1),:]
    # transformed_cloud_o3d = o3d.geometry.PointCloud()
    # transformed_cloud_o3d.points = o3d.utility.Vector3dVector(transformed_cloud_np)
    # transformed_cloud_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # if isVoxelDown:
    #     transformed_cloud_o3d = transformed_cloud_o3d.voxel_down_sample(voxel_size=2e-6)

    if not os.path.exists(cloud_save_dir):
        os.makedirs(cloud_save_dir)
    filename = img_path.split('/')[-1].split('.')[0]+'.ply'
    o3d.io.write_point_cloud(os.path.join(cloud_save_dir,filename), transformed_cloud_o3d, 
                                write_ascii=False, compressed=False, print_progress=False)



    # o3d.visualization.draw_geometries([voxel_down_pcd])

    """## Mesh generation"""
    if isMeshGen:
        transformed_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0104,max_nn=15))
        print('run Poisson surface reconstruction')
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(transformed_cloud_o3d, depth=15, width=0, scale=2.1, linear_fit=True)


    """## Volume estimation"""
    if isVolEst:
        xyz = np.asarray(transformed_cloud_o3d.points)
        xy_catalog = []
        for point in xyz:
            xy_catalog.append([point[0], point[1]])
        tri = Delaunay(np.array(xy_catalog))

        xyz[0]

        xy_catalog = np.array(xy_catalog)
        # plt.triplot(xy_catalog[:,0], xy_catalog[:,1], tri.simplices)

        surface = o3d.geometry.TriangleMesh()
        surface.vertices = o3d.utility.Vector3dVector(xyz)
        surface.triangles = o3d.utility.Vector3iVector(tri.simplices)

        volume = reduce(lambda a, b:  a + volume_under_triangle(b), get_triangles_vertices(surface.triangles, surface.vertices), 0)
        print(f"The volume is: {round(volume, 4)} m3")
    
    return depth_calibration_pipeline, transformed_cloud_o3d

