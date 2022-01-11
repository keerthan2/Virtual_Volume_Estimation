from imports import *
from utils import *

def linear_plane_estimation(points):
  """Find the plane that best fits the input points using ordinary 
  least squares.
  Inputs:
      points: Input points Mx3.
  Outputs:
      params: Plane parameters (w0,w1,w2,w3).
  """
  # Use OLS to estimate the plane parameters
  model = linear_model.LinearRegression()
  model.fit(points[:,:2], points[:,2:])
  params = (model.intercept_.tolist() 
            + model.coef_[0].tolist() + [-1])
  return params

def pca_plane_estimation(points):
  """Find the plane that best fits the input poitns using PCA.
  Inputs:
      points: Input points Mx3.
  Returns:
      params: Plane parameters (w0,w1,w2,w3).
  """
  # Fit a linear model to determine the plane normal orientation
  linear_params = linear_plane_estimation(points)
  linear_model_normal = np.array(linear_params[1:])

  # Zero mean the points and compute the covariance
  # matrix eigenvalues/eigenvectors
  point_mean = np.mean(points, axis=0)
  zero_mean_points = points - point_mean
  cov_matrix = np.cov(zero_mean_points.T)
  eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
  # Sort the eigenvectors in descending eigenvalue orded
  sort_indices = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[sort_indices]
  eigenvectors = eigenvectors[:, sort_indices]

  # Use the least important component's eigenvector as plane normal
  normal = eigenvectors[:,2]
  # Align the PCA plane normal with the linear model plane normal
  if np.dot(normal, linear_model_normal) < 0:
      normal = normal * (-1)
  params = [-np.dot(normal, point_mean), normal[0], normal[1], normal[2]]

  return params

def align_plane_with_axis(plane_params, axis):
  """Compute the translation vector and rotation matrix to align 
  plane normal with axis.
  
  Inputs:
      plane_params: Plane parameters (w0,w1,w2,w3).
      axis: Unit axis to align plane to.
  Returns:
      translation_vec: Translation vector.
      rotation_matrix: Rotation matrix.
  """
  plane_params = np.array(plane_params)
  plane_normal = plane_params[1:] / np.sqrt(np.sum(plane_params[1:]**2))
  # Compute translation vector
  d = plane_params[0] / (np.dot(plane_normal, axis) 
                          * np.sqrt(np.sum(plane_params[1:]**2)))
  translation_vec = d * axis
  # Compute rotation matrix
  rot_axis = np.cross(plane_normal, axis)
  rot_axis_norm = rot_axis / np.sqrt(np.sum(rot_axis**2))
  angle = np.arccos(np.dot(plane_normal, axis))
  r = Rotation.from_rotvec(angle * rot_axis_norm)
  rotation_matrix = r.as_dcm()
  return translation_vec, rotation_matrix

def sor_filter(points, z_max=1, inlier_ratio=0.5):
  """Statistical outlier filtering of point cloud data.
  Inputs:
      points: Input points Mx3.
      z_max: Maximum z-score for inliers.
      inlier_ratio: Assumption of min inliers to outliers ratio.
  Returns:
      inliers: Inlier points in input set.
      sor_mask: Mask of inlier points.
  """
  # Find max k-neighbor distance to use as distance score
  # where k is determined by the assumed inlier to outlier ratio
  kdtree = cKDTree(points)
  k = inlier_ratio * points.shape[0]
  distances, _ = kdtree.query(points, k)
  z_scores = zscore(np.max(distances, axis=1))
  # Filter out points outside given z-score range
  sor_mask = np.abs(z_scores) < z_max
  inliers = points[sor_mask]
  return inliers, sor_mask

def get_triangles_vertices(triangles, vertices):
    triangles_vertices = []
    for triangle in triangles:
        new_triangles_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
        triangles_vertices.append(new_triangles_vertices)
    return np.array(triangles_vertices)

def volume_under_triangle(triangle):
    p1, p2, p3 = triangle
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    return abs((z1+z2+z3)*(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)/6)

def obj2cloud(depth_calibration_pipeline, obj_name, depth_save_dir = "depth_output", intrinsics_mat_path='cameraIntrinsic.xml', voxel_size=0.0000007):
  # fov = 70 # have to write a function
  input_image_shape = depth_calibration_pipeline.segmentation_img.shape
  img_path = depth_calibration_pipeline.img_path
  fname = img_path.split('/')[-1]
  depth_path = f'./{depth_calibration_pipeline.depth_save_dir}/{fname}'
  depth_pfm_path = f"./{depth_save_dir}/{fname.split('.')[0]}"+'.pfm'
  
  img = cv2.imread(img_path)
  color = o3d.geometry.Image(img)

  if intrinsics_mat_path is not None:
    cm_file = cv2.FileStorage()
    cm_file.open(intrinsics_mat_path,cv2.FileStorage_READ)
    camera_intrinsic_matrix = cm_file.getNode('intrinsic').mat()
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, 
                                                        fx=camera_intrinsic_matrix[0][0], 
                                                        fy=camera_intrinsic_matrix[1][1], 
                                                        cx=camera_intrinsic_matrix[0][-1], 
                                                        cy=camera_intrinsic_matrix[1][-1])
  else:
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

  idepth = read_pfm(depth_pfm_path)[0]
  idepth = idepth - np.amin(idepth)
  idepth /= np.amax(idepth)
  focal = camera_intrinsic.intrinsic_matrix[0, 0]
  depth = focal / (idepth)
  depth = o3d.geometry.Image(depth)

  rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=focal, depth_trunc=5)
  
  # Create the point cloud from images and camera intrisic parameters
  pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic, project_valid_depth_only=False)
  pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
  voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
  voxel_down_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
  point_cloud_flat = np.asarray(pcd.points)
  # print(point_cloud.shape)
  # point_cloud_flat = np.reshape(point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3)) # BS x HW x 3

  obj_idx = depth_calibration_pipeline.segmentation_results['class_names'].index(obj_name)
  obj_mask = depth_calibration_pipeline.segmentation_results['masks'][:,:,obj_idx]
  # obj_scaled_depth_map = obj_mask*depth_calibration_pipeline.scaled_depth_map
  obj_scaled_depth_map = obj_mask*depth_calibration_pipeline.depth_map

  # img = Image.fromarray(np.uint8(obj_scaled_depth_map)).convert('RGB')
  # display(img)
  # plt.imshow(obj_scaled_depth_map)
  # plt.colorbar(label="scaled depth (cm)", orientation="vertical")
  # plt.show()
  obj_mask = (np.reshape(
                obj_scaled_depth_map, (obj_scaled_depth_map.shape[0] * obj_scaled_depth_map.shape[1])) > 0
                 )
  obj_points = point_cloud_flat[obj_mask, :]
  # print(obj_points.shape)
  non_obj_points = point_cloud_flat[np.logical_not(obj_mask), :]
  obj_points_df = pd.DataFrame(obj_points, columns=['x','y','z'])
  obj_cloud = PyntCloud(obj_points_df)
  
  # obj_points_filtered, sor_mask = sor_filter(obj_points, 2, 0.7)
  # obj_points_filtered = obj_points
  # plane_params = pca_plane_estimation(obj_points_filtered)
  # translation, rotation_matrix = align_plane_with_axis(plane_params, np.array([0, 0, 1]))
  
  # obj_points_transformed = np.dot(obj_points_filtered + translation, rotation_matrix.T)
  # obj_points_transformed_df = pd.DataFrame(obj_points_transformed, columns=['x','y','z'])
  # obj_transformed_cloud = PyntCloud(obj_points_transformed_df)

  return obj_cloud#, obj_transformed_cloud 