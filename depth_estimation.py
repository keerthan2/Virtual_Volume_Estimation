from imports import *
from utils import *

class DepthEstimator:
  
  def __init__(self, model_path, model_type="large", optimize=False):    
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
        # if self.optimize and self.device == torch.device("cuda"):
        #     sample = sample.to(memory_format=torch.channels_last)  
        #     sample = sample.half()
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


class DepthCalibrationPipeline:
  def __init__(self, segmentation_model_path, depth_model_base_path, depth_model_type = "dpt_hybrid", depth_optimize = False):
  
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
    self.scaled_depth_map = None
  
  def segmentor(self, img_path):
    return self.ins.segmentImage(img_path, show_bboxes = True, extract_segmented_objects = True)
  
  def get_depth_map(self, img_path):    
    predictions = self.depth_model.run(img_path)
    predictions_minmax = (predictions-np.min(predictions))/np.max(predictions)
    depth_map = 1-predictions_minmax
    return predictions, depth_map
  
  def get_scaled_depthmap(self, segmentation_results, depth_map, ref_obj_name, ref_gt_depth):
    ref_obj_name = "bottle" # should be present in class_names in predictions
    ref_obj_idx = segmentation_results['class_names'].index(ref_obj_name)
    ref_obj_mask = segmentation_results['masks'][:,:,ref_obj_idx]
    ref_obj_depth_map = ref_obj_mask*depth_map
    ref_obj_area, ref_depth_area = np.sum(ref_obj_mask), np.sum(ref_obj_depth_map)
    ref_avg_depth = ref_depth_area/ref_obj_area
    scale = ref_gt_depth/ref_avg_depth
    scaled_depth_map = scale*depth_map
    return scaled_depth_map

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

"""## Initialize pipeline"""
if __name__ == '__main__':
    segmentation_model_path = "./segmentation_model/pointrend_resnet50.pkl"
    depth_model_base_path = './midas_depth/weights/'
    depth_calibration_pipeline = DepthCalibrationPipeline(segmentation_model_path, depth_model_base_path)

    """## Run base pipeline to get depth estimates"""

    img_path = "imgs/rice_example.jpeg"
    depth_save_dir = "depth_output"
    depth_calibration_pipeline.run_base_block(img_path, depth_save_dir)

    # depth_calibration_pipeline.midas_pred

    arr = depth_calibration_pipeline.segmentation_img
    img = Image.fromarray(np.uint8(arr)).convert('RGB')
    print(img.size)
    display(img)
    print(depth_calibration_pipeline.segmentation_results['class_names'])