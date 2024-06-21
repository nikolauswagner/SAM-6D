#import rospy
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
#from hydra import initialize, compose
#from hydra.utils import instantiate
import hydra
import trimesh
import time
import argparse
import glob
from omegaconf import OmegaConf
import torch
import cv2
import imageio.v2 as imageio
import distinctipy
import gorilla
import random
import importlib
import json
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import pycocotools.mask as cocomask

# Instance segmentation tools
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23

from Pose_Estimation_Model.utils.data_utils import load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose

obj_names = [ "001_chips_can",
              "002_master_chef_can",
              "003_cracker_box",
              "004_sugar_box",
              "005_tomato_soup_can",
              "006_mustard_bottle",
              "007_tuna_fish_can",
              "008_pudding_box",
              "009_gelatin_box",
              "010_potted_meat_can",
              "011_banana",
              "012_strawberry",
              "013_apple",
              "014_lemon",
              "015_peach",
              "016_pear",
              "017_orange",
              "018_plum",
              "021_bleach_cleanser",
              "024_bowl",
              "025_mug",
              "029_plate",
              "030_fork",
              "031_spoon",
              "032_knife",
              "053_mini_soccer_ball",
              "054_softball",
              "055_baseball",
              "056_tennis_ball",
              "057_racquetball",
              "058_golf_ball",
              "062_dice",
              "077_rubiks_cube"]

rgb_transform = T.Compose([T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])


def batch_input_data(depth_path, cam_info, device):
    batch = {}
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def visualize_ism(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    for mask_idx, det in enumerate(detections):
      if mask_idx == 3:
        break
      mask = rle_to_mask(det["segmentation"])
      edge = canny(mask)
      edge = binary_dilation(edge, np.ones((2, 2)))
      obj_id = det["category_id"]
      temp_id = obj_id - 1

      r = int(255*colors[temp_id][0])
      g = int(255*colors[temp_id][1])
      b = int(255*colors[temp_id][2])
      img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
      img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
      img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
      img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat


class SAM6DRunner(object):

  def __init__(self, 
               cad_dir="/home/niko/Documents/data/ycb/models/", 
               template_dir="/home/niko/Documents/data/ycb/templates/",
               ism_type="fastsam"):
    print("Initialising SAM-6D...")
    time_0 = time.time()
    self.template_dir = template_dir
    self.cad_dir = cad_dir
    self.ism_type = ism_type

    self.cam_info = load_json("/home/niko/Documents/git/SAM-6D/SAM-6D/Data/Example/camera.json")
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.meshes = []
    self.templates = []

    # Load models
    self.load_model_segmentation()
    self.load_model_pose_estimation()

    # Load data
    self.load_cad_models()
    self.load_templates_segmentation()
    self.load_templates_pose_estimation()
    time_1 = time.time()
    print("Initialising SAM-6D took {:.3f}s.".format(time_1 - time_0))

  def load_cad_models(self):
    self.mesh = trimesh.load_mesh(self.cad_dir + "013_apple" + "/textured.obj")
    return

  def load_templates_segmentation(self, obj_name="001_chips_can"):
    template_dir = os.path.join(self.template_dir, obj_name)
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
      image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
      mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
      boxes.append(mask.getbbox())

      image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
      mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
      image = image * mask[:, :, None]
      templates.append(image)
      masks.append(mask.unsqueeze(-1))

    for idx in range(num_templates):
      image = Image.open(os.path.join(template_dir + "/../002_master_chef_can", 'rgb_'+str(idx)+'.png'))
      mask = Image.open(os.path.join(template_dir + "/../002_master_chef_can", 'mask_'+str(idx)+'.png'))
      boxes.append(mask.getbbox())

      image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
      mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
      image = image * mask[:, :, None]
      templates.append(image)
      masks.append(mask.unsqueeze(-1))

    for idx in range(num_templates):
      image = Image.open(os.path.join(template_dir + "/../003_cracker_box", 'rgb_'+str(idx)+'.png'))
      mask = Image.open(os.path.join(template_dir + "/../003_cracker_box", 'mask_'+str(idx)+'.png'))
      boxes.append(mask.getbbox())

      image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
      mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
      image = image * mask[:, :, None]
      templates.append(image)
      masks.append(mask.unsqueeze(-1))

    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))

    processing_config = OmegaConf.create({"image_size": 224})
    proposal_processor = CropResizePad(processing_config.image_size)
    self.templates_ism = proposal_processor(images=templates, boxes=boxes).to(self.device)
    self.masks_cropped_ism = proposal_processor(images=masks, boxes=boxes).to(self.device)

  def load_templates_pose_estimation(self, obj_name="001_chips_can"):
    self.all_tem = []
    self.all_tem_choose = []
    self.all_tem_pts = []

    for i in range(42):
      rgb = load_im(self.template_dir + obj_name + "/rgb_{:d}.png".format(i)).astype(np.uint8)
      mask = load_im(self.template_dir + obj_name + "/mask_{:d}.png".format(i)).astype(np.uint8) == 255
      xyz = np.load(self.template_dir + obj_name + "/xyz_{:d}.npy".format(i)).astype(np.float32) / 1000.0  
      
      bbox = get_bbox(mask)
      y1, y2, x1, x2 = bbox
      mask = mask[y1:y2, x1:x2]

      rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
      rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

      rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
      rgb = rgb_transform(np.array(rgb))

      choose = (mask > 0).astype(np.float32).flatten().nonzero()[0]
      n_sample_template_point = 5000
      if len(choose) <= n_sample_template_point:
          choose_idx = np.random.choice(np.arange(len(choose)), n_sample_template_point)
      else:
          choose_idx = np.random.choice(np.arange(len(choose)), n_sample_template_point, replace=False)
      choose = choose[choose_idx]
      xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

      rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], 224)

      self.all_tem.append(torch.FloatTensor(rgb).unsqueeze(0).cuda())
      self.all_tem_choose.append(torch.IntTensor(rgb_choose).long().unsqueeze(0).cuda())
      self.all_tem_pts.append(torch.FloatTensor(xyz).unsqueeze(0).cuda())

    with torch.no_grad():
      self.all_tem_pts, self.all_tem_feat = self.model_posest.feature_extraction.get_obj_feats(self.all_tem, self.all_tem_pts, self.all_tem_choose)

  def load_model_segmentation(self):
    with hydra.initialize(version_base=None, config_path="Instance_Segmentation_Model/configs"):
      self.cfg_ism = hydra.compose(config_name='run_inference.yaml')
    if self.ism_type == "fastsam":
      with hydra.initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
        self.cfg_ism.model = hydra.compose(config_name='ISM_fastsam.yaml')
    #else:
    #  rospy.logerr("ISM model not supported!")

    self.model_ism = hydra.utils.instantiate(self.cfg_ism.model)

    self.model_ism.descriptor_model.model = self.model_ism.descriptor_model.model.to(self.device)
    self.model_ism.descriptor_model.model.device = self.device

  def load_model_pose_estimation(self):
    random.seed(1)
    torch.manual_seed(1)

    gorilla.utils.set_cuda_visible_devices(gpu_ids=0)
    cfg = gorilla.Config.fromfile("Pose_Estimation_Model/config/base.yaml")

    # model
    MODEL = importlib.import_module("Pose_Estimation_Model.model.pose_estimation_model")
    self.model_posest = MODEL.Net(cfg.model)
    self.model_posest = self.model_posest.cuda()
    self.model_posest.eval()
    checkpoint = "./Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth"
    gorilla.solver.load_checkpoint(model=self.model_posest, filename=checkpoint)

  def detect_object(self, img):
    self.model_ism.ref_data = {}
    self.model_ism.ref_data["descriptors"] = self.model_ism.descriptor_model.compute_features(
                    self.templates_ism, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    self.model_ism.ref_data["appe_descriptors"] = self.model_ism.descriptor_model.compute_masked_patch_feature(
                    self.templates_ism, self.masks_cropped_ism[:, 0, :, :]
                ).unsqueeze(0).data
    
    # run inference
    print(time.time())
    time_0 = time.time()
    detections = self.model_ism.segmentor_model.generate_masks(np.array(img))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = self.model_ism.descriptor_model.forward(np.array(img), detections)
    time_1 = time.time()
    print("Took: {:.3f}".format(time_1 - time_0))

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = self.model_ism.compute_semantic_score(query_decriptors)

    # Update detection
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor = self.model_ism.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score
    depth_path = "/home/niko/Documents/git/SAM-6D/SAM-6D/Data/Example/depth.png"
    batch = batch_input_data(depth_path, self.cam_info, self.device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
    self.model_ism.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

    model_points = self.mesh.sample(2048).astype(np.float32) / 1000.0
    self.model_ism.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(self.device)

    print(best_template, pred_idx_objects, batch, detections.masks)    
    image_uv = self.model_ism.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = self.model_ism.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.model_ism.visible_thred
        )

    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
    detections.to_numpy()
    save_path = "/home/niko/Documents/data/ycb/sam6d_results/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    self.detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])

    #save_json_bop23(save_path+".json", detections)
    vis_img = visualize_ism(img, self.detections, "/home/niko/Documents/data/ycb/sam6d_results/vis_ism.png")
    #vis_img.save("/home/niko/Documents/data/ycb/sam6d_results//vis_ism.png")

  def detect_pose(self, det_score_thresh=0.2):
    dets = []
    for det in self.detections:
      if det['score'] > det_score_thresh:
          dets.append(det)

    K = np.array(self.cam_info['cam_K']).reshape(3, 3)

#    whole_image = load_im(rgb_path).astype(np.uint8)
#    if len(whole_image.shape)==2:
#        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
#    whole_depth = load_im(depth_path).astype(np.float32) * self.cam_info['depth_scale'] / 1000.0
#    whole_pts = get_point_cloud_from_depth(whole_depth, K)
#
#    n_sample_model_point = 1024
#    model_points = self.mesh.sample(n_sample_model_point).astype(np.float32) / 1000.0
#    radius = np.max(np.linalg.norm(model_points, axis=1))
#
#
#    all_rgb = []
#    all_cloud = []
#    all_rgb_choose = []
#    all_score = []
#    all_dets = []
#    for inst in dets:
#        seg = inst['segmentation']
#        score = inst['score']
#
#        # mask
#        h,w = seg['size']
#        try:
#            rle = cocomask.frPyObjects(seg, h, w)
#        except:
#            rle = seg
#        mask = cocomask.decode(rle)
#        mask = np.logical_and(mask > 0, whole_depth > 0)
#        if np.sum(mask) > 32:
#            bbox = get_bbox(mask)
#            y1, y2, x1, x2 = bbox
#        else:
#            continue
#        mask = mask[y1:y2, x1:x2]
#        choose = mask.astype(np.float32).flatten().nonzero()[0]
#
#        # pts
#        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
#        center = np.mean(cloud, axis=0)
#        tmp_cloud = cloud - center[None, :]
#        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
#        if np.sum(flag) < 4:
#            continue
#        choose = choose[flag]
#        cloud = cloud[flag]
#
#        if len(choose) <= cfg.n_sample_observed_point:
#            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
#        else:
#            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
#        choose = choose[choose_idx]
#        cloud = cloud[choose_idx]
#
#        # rgb
#        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
#        if cfg.rgb_mask_flag:
#            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
#        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
#        rgb = rgb_transform(np.array(rgb))
#        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
#
#        all_rgb.append(torch.FloatTensor(rgb))
#        all_cloud.append(torch.FloatTensor(cloud))
#        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
#        all_score.append(score)
#        all_dets.append(inst)
#
#    ret_dict = {}
#    ret_dict['pts'] = torch.stack(all_cloud).cuda()
#    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
#    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
#    ret_dict['score'] = torch.FloatTensor(all_score).cuda()
#
#    ninstance = ret_dict['pts'].size(0)
#    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
#    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
#    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets


if __name__ == '__main__':
  s6d = SAM6DRunner()
  img = Image.open("/home/niko/Documents/data/robocup/shelf.png").convert("RGB")
  while True:
    s6d.detect_object(img)
  print("All done")