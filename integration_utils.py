from imports import *

class DepthEstimator:
  
  def __init__(self, model_path, model_type="large", optimize=True):    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.optimize = optimize
    if model_type == "dpt_large": # DPT-Large
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        self.net_w, self.net_h = 384, 384
        self.resize_mode = "minimal"
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        self.net_w, self.net_h = 384, 384
        self.resize_mode="minimal"
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        self.model = MidasNet(model_path, non_negative=True)
        self.net_w, self.net_h = 384, 384
        self.resize_mode="upper_bound"
        self.normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        self.model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        self.net_w, self.net_h = 256, 256
        self.resize_mode="upper_bound"
        self.normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    self.transform = Compose(
                      [
                          Resize(
                              self.net_w,
                              self.net_h,
                              resize_target=None,
                              keep_aspect_ratio=True,
                              ensure_multiple_of=32,
                              resize_method=self.resize_mode,
                              image_interpolation_method=cv2.INTER_CUBIC,
                          ),
                          self.normalization,
                          PrepareForNet(),
                      ]
                  )
    
    self.model.eval()

    if self.optimize:
        if self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)  
            self.model = self.model.half()

    self.model.to(self.device)

  def read_image(self, path):
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img
  
  def post_process_depth(self, depth, bits=2):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)
    return out

  def run(self, input_path):
    img_name = input_path
    img = self.read_image(img_name)
    img_input = self.transform({"image": img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
        if self.optimize and self.device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)  
            sample = sample.half()
        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    # prediction = self.post_process_depth(prediction)
    return prediction

def read_pfm(path):
    """Read pfm file.
    Args:
        path (str): path to file
    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:
      color = None
      width = None
      height = None
      scale = None
      endian = None

      header = file.readline().rstrip()
      if header.decode("ascii") == "PF":
          color = True
      elif header.decode("ascii") == "Pf":
          color = False
      else:
          raise Exception("Not a PFM file: " + path)

      dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
      if dim_match:
          width, height = list(map(int, dim_match.groups()))
      else:
          raise Exception("Malformed PFM header.")

      scale = float(file.readline().decode("ascii").rstrip())
      if scale < 0:
          # little-endian
          endian = "<"
          scale = -scale
      else:
          # big-endian
          endian = ">"

      data = np.fromfile(file, endian + "f")
      shape = (height, width, 3) if color else (height, width)

      data = np.reshape(data, shape)
      data = np.flipud(data)

      return data, scale


def write_pfm(path, image, scale=1):
  """Write pfm file.
  Args:
      path (str): pathto file
      image (array): data
      scale (int, optional): Scale. Defaults to 1.
  """
  with open(path, "wb") as file:
      color = None

      if image.dtype.name != "float32":
          raise Exception("Image dtype must be float32.")

      image = np.flipud(image)

      if len(image.shape) == 3 and image.shape[2] == 3:  # color image
          color = True
      elif (
          len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
      ):  # greyscale
          color = False
      else:
          raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

      file.write("PF\n" if color else "Pf\n".encode())
      file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

      endian = image.dtype.byteorder

      if endian == "<" or endian == "=" and sys.byteorder == "little":
          scale = -scale

      file.write("%f\n".encode() % scale)

      image.tofile(file)

class DepthCalibrationPipeline:
  def __init__(self, segmentation_model_path, depth_model_base_path, depth_model_type = "dpt_hybrid", depth_optimize = True):
  
    self.segmentation_model_path = segmentation_model_path
    self.ins = instanceSegmentation()
    self.ins.load_model(self.segmentation_model_path)
    
    self.depth_model_base_path = depth_model_base_path
    self.depth_model_type = depth_model_type
    self.depth_optimize = depth_optimize
    default_models = {
            "midas_v21_small": os.path.join(depth_model_base_path,"midas_v21_small-70d6b9c8.pt"),
            "midas_v21": os.path.join(depth_model_base_path,"/midas_v21-f6b98070.pt"),
            "dpt_large": os.path.join(depth_model_base_path,"dpt_large-midas-2f21e586.pt"),
            "dpt_hybrid": os.path.join(depth_model_base_path,"dpt_hybrid-midas-501f0c75.pt"),
        }
    self.depth_model_weights = default_models[depth_model_type]
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    self.depth_model = DepthEstimator(self.depth_model_weights, model_type=depth_model_type, optimize=depth_optimize)

    self.segmentation_results = None
    self.depth_map = None
    self.depth_scale = None
    self.scaled_depth_map = None
    self.obj_req_data = {'id':None, 'name':None, 'box':None, 'mask':None}

  def segmentor(self, img_path):
    return self.ins.segmentImage(img_path, show_bboxes = True, extract_segmented_objects = True)
  
  def get_depth_map(self, img_path):    
    predictions = self.depth_model.run(img_path)
    predictions_minmax = (predictions-np.min(predictions))/np.max(predictions)
    depth_map = 1-predictions_minmax
    return predictions, depth_map
  
  def get_scaled_depthmap_v1(self, segmentation_results, depth_map, ref_obj_name, ref_gt_depth): # reference object based scaling
    ref_obj_name = "bottle" # should be present in class_names in predictions
    ref_obj_idx = segmentation_results['class_names'].index(ref_obj_name)
    ref_obj_mask = segmentation_results['masks'][:,:,ref_obj_idx]
    ref_obj_depth_map = ref_obj_mask*depth_map
    ref_obj_area, ref_depth_area = np.sum(ref_obj_mask), np.sum(ref_obj_depth_map)
    ref_avg_depth = ref_depth_area/ref_obj_area
    scale = ref_gt_depth/ref_avg_depth
    scaled_depth_map = scale*depth_map
    return scaled_depth_map
  
  def get_scaled_depthmap(self, gt_background_depth, margin = 50, centered=1): # background based scaling
    def compute_area(xl, yl, xr, yr):
      return abs((xr-xl)*(yr-yl))
    H,W,C = self.segmentation_img.shape
    xc, yc = W//2, H//2
    min_area = float('inf')
    for i, box in enumerate(self.segmentation_results['boxes']):
      xl, yl, xr, yr = box
      area = compute_area(xl, yl, xr, yr)
      if centered:
        if (((xl<=xc<=xr) and (yl<=yc<=yr)) or \
            ((self.obj_req_data['id']!=None) and \
            (self.obj_req_data['id'] == self.segmentation_results['class_ids'][i]))) and \
            (area < min_area):
          self.obj_req_data['id'] = self.segmentation_results['class_ids'][i]
          self.obj_req_data['name'] = self.segmentation_results['class_names'][i]
          self.obj_req_data['box'] = box
          self.obj_req_data['mask'] = self.segmentation_results['masks'][:,:,i]
          min_area = area
      else:
        self.obj_req_data['id'] = self.segmentation_results['class_ids'][i]
        self.obj_req_data['name'] = self.segmentation_results['class_names'][i]
        self.obj_req_data['box'] = box
        self.obj_req_data['mask'] = self.segmentation_results['masks'][:,:,i]
        min_area = area
    
    xl, yl, xr, yr = self.obj_req_data['box']
    # xml, yml, xmr, ymr = max(xl-margin,0), max(yl-margin,0), min(xr+margin,W-1), min(yr+margin,H-1)
    xml, yml, xmr, ymr = max(xl-2*margin,0), max(yl-2*margin,0), min(xr-margin,W-1), min(yr-margin,H-1)
    background_mask = np.zeros((H,W))
    background_mask[yml:ymr,xml:xmr] = 1
    background_mask[yl:yr,xl:xr] = 0
    background_depth_map = background_mask*self.depth_map

    area_background, area_obj = compute_area(xml, yml, xmr, ymr), compute_area(xl, yl, xr, yr)
    area_margin = area_background-area_obj
    total_background_depth = np.sum(background_depth_map)
    avg_background_depth = total_background_depth/area_margin

    self.depth_scale = gt_background_depth/avg_background_depth
    self.scaled_depth_map = self.depth_scale*self.depth_map

    return self.depth_scale, self.scaled_depth_map


  def obj2depth(self, scaled_depth_map, results, obj_name):
    obj_idx = results['class_names'].index(obj_name)
    obj_mask = results['masks'][:,:,obj_idx]
    obj_scaled_depth_map = obj_mask*scaled_depth_map
    obj_area = np.sum(obj_mask)
    depth_area = np.sum(obj_scaled_depth_map)
    avg_depth = depth_area/obj_area
    return avg_depth
  
  def run_base_block(self, img_path, depth_save_dir=None):
    self.img_path = img_path
    self.depth_save_dir = depth_save_dir
    segmentor_output = self.segmentor(img_path)
    self.segmentation_results = segmentor_output[0]
    self.segmentation_img = segmentor_output[1]
    self.midas_pred, self.depth_map = self.get_depth_map(img_path)
    if depth_save_dir is not None:
      processed_depth_map = self.depth_model.post_process_depth(self.midas_pred, bits=1)
      # depth_img = Image.fromarray(255-np.uint8(self.depth_map*255)).convert('RGB')
      fname = img_path.split('/')[-1]
      cv2.imwrite(f'./{depth_save_dir}/{fname}', processed_depth_map.astype("uint8"))
      write_pfm(f"./{depth_save_dir}/{fname.split('.')[0]}"+'.pfm', self.midas_pred.astype(np.float32))

  def run_scaler(self, ref_obj_name, ref_gt_depth):
    self.scaled_depth_map = self.get_scaled_depthmap(self.segmentation_results, self.depth_map, ref_obj_name, ref_gt_depth)

  def run_without_base(self, obj_name, ref_obj_name, ref_gt_depth):
    self.scaled_depth_map = self.get_scaled_depthmap(self.segmentation_results, self.depth_map, ref_obj_name, ref_gt_depth)
    return self.obj2depth(self.scaled_depth_map, self.segmentation_results, obj_name)

  def run_full_pipeline(self, img_path, obj_name, ref_obj_name, ref_gt_depth):
    self.run_base_block(img_path)
    return self.run_without_base(obj_name, ref_obj_name, ref_gt_depth)
  
  def run_only_on_obj(self, obj_name):
    return self.obj2depth(self.scaled_depth_map, self.segmentation_results, obj_name)

def id2pc(img_path, depth_pfm_path, cam_mat_path, cloud_save_dir, depth_scale):

    img = cv2.imread(img_path)
    color = o3d.geometry.Image(img)
    if cam_mat_path is not None:
        cm_file = cv2.FileStorage()
        cm_file.open(cam_mat_path,cv2.FileStorage_READ)
        camera_intrinsic_matrix = cm_file.getNode('intrinsic').mat()
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, 
                                                            fx=camera_intrinsic_matrix[0][0], 
                                                            fy=camera_intrinsic_matrix[1][1], 
                                                            cx=camera_intrinsic_matrix[0][-1], 
                                                            cy=camera_intrinsic_matrix[1][-1])
    else:
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    # print(camera_intrinsic.intrinsic_matrix)

    idepth = read_pfm(depth_pfm_path)[0]
    idepth = idepth - np.amin(idepth)
    idepth /= np.amax(idepth)

    depth = (1 - idepth)+1e-6
    
    depth = o3d.geometry.Image(depth)

    idepth_scale = 1/depth_scale

    rgbdi = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=idepth_scale, depth_trunc=1000)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbdi, camera_intrinsic, project_valid_depth_only=True)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if cloud_save_dir is not None:
        if not os.path.exists(cloud_save_dir):
            os.makedirs(cloud_save_dir)
        filename = img_path.split('/')[-1].split('.')[0]+'.ply'
        o3d.io.write_point_cloud(os.path.join(cloud_save_dir,filename), pcd, 
                                write_ascii=False, compressed=False, print_progress=False)
    return pcd

def obj2cloud(depth_calibration_pipeline, gt_background_depth, intrinsics_mat_path='cameraIntrinsic.xml', cloud_save_dir='pointclouds', margin=50, centered=1, voxel_size=0.0000007):
  input_image_shape = depth_calibration_pipeline.segmentation_img.shape
  img_path = depth_calibration_pipeline.img_path
  fname = img_path.split('/')[-1]
  depth_path = f'./{depth_calibration_pipeline.depth_save_dir}/{fname}'
  depth_pfm_path = f"./{depth_calibration_pipeline.depth_save_dir}/{fname.split('.')[0]}"+'.pfm'
  
  img = cv2.imread(img_path)
  color = o3d.geometry.Image(img)

  depth_scale, _ = depth_calibration_pipeline.get_scaled_depthmap(gt_background_depth=gt_background_depth, margin=margin, centered=centered)
  
  pcd = id2pc(img_path, depth_pfm_path, intrinsics_mat_path, cloud_save_dir, depth_scale)
  point_cloud_flat = np.asarray(pcd.points)
  obj_mask = depth_calibration_pipeline.obj_req_data['mask']
  obj_scaled_depth_map = obj_mask*depth_calibration_pipeline.depth_map
  obj_mask = (np.reshape(
                obj_scaled_depth_map, (obj_scaled_depth_map.shape[0] * obj_scaled_depth_map.shape[1])) > 0
                 )
  obj_points = point_cloud_flat[obj_mask, :]
  non_obj_points = point_cloud_flat[np.logical_not(obj_mask), :]
  obj_points_df = pd.DataFrame(obj_points, columns=['x','y','z'])
  obj_cloud = PyntCloud(obj_points_df)

  transformed_cloud = obj_cloud
  transformed_cloud_np = transformed_cloud.points.to_numpy()
  transformed_cloud_np = transformed_cloud_np[~np.isnan(transformed_cloud_np).any(axis=1),:]
  transformed_cloud_o3d = o3d.geometry.PointCloud()
  transformed_cloud_o3d.points = o3d.utility.Vector3dVector(transformed_cloud_np)
  transformed_cloud_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

  return transformed_cloud_o3d

def compute_volume(pcd):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(np.asarray(pcd.points))
    return hull.volume