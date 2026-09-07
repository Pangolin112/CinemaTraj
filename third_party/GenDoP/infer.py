import os
import cv2
import tyro
import glob
import time
import json
import math
import shutil
import numpy as np
import torch
from PIL import Image
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import kiui
import trimesh
from kiui.op import recenter

from core.options import AllConfigs, Options
from core.models import LMM
from core.utils import monkey_patch_transformers
from core.utils import camera_to_token, camera_to_token_single, token_to_camera, quaternion_to_matrix, matrix_to_quaternion, quaternion_slerp, sample_from_two_pose, sample_from_dense_cameras
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt

monkey_patch_transformers()

opt = tyro.cli(AllConfigs)

kiui.seed_everything(opt.seed)

# model
model = LMM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sane?')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().eval().to(device)

def draw_json(c2ws, vis_path):
    output_dir = os.path.dirname(vis_path)
    parent_dir = os.path.dirname(output_dir)

    rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3]))) * 1.1

    # Prepare visualizer
    visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])
    num_matrices = c2ws.shape[0]

    # Create a color gradient from red to purple
    colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))

    # Create three views
    views = [
        {'elev': 90, 'azim': -90, 'name': 'front'},
        {'elev': 180, 'azim': -90, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'side'}
    ]
    
    image_paths = []

    for view in views:
        fig = plt.figure(figsize=(12, 12))  # Each image will be 4x12 inches
        visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

        for i in range(num_matrices):
            color = colors[i]
            visualizer.extrinsic2pyramid(c2ws[i], color, rangesize / 4)
        
        visualizer.ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Save each view as a separate image
        image_path = f"{parent_dir}/{view['name']}_view.png"
        os.makedirs(output_dir, exist_ok=True)
        visualizer.save(image_path)
        image_paths.append(image_path)
    
    # Now combine the three images horizontally
    images = [Image.open(img_path) for img_path in image_paths]
    images[-1] = images[-1].rotate(90, expand=True)

    # Resize images to fit the desired final size
    images = [img.crop((420, 420, 1980, 1980)) for img in images]
    images_resized = [img.resize((341, 341)) for img in images]

    # Concatenate images horizontally
    combined_image = np.concatenate([np.array(img) for img in images_resized], axis=1)

    # Save the final combined image
    final_image = Image.fromarray(combined_image)
    final_image.save(vis_path)

    print(f"Combined image saved at {vis_path}")


def process_data(opt, output_dir, name, image_path, text_path=None, depth_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    new_traj_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
    if os.path.exists(new_traj_path):
        print(f"Skipping {name} as it already exists.")
        return

    text = None
    if text_path is not None:
        info = json.load(open(text_path, 'r'))
        if opt.cond_mode == 'text':
            text_key = opt.text_key
        else:
            text_key = 'Concise Interaction'
        text = info[text_key]

    if opt.cond_mode == 'text':
        assert text_path is not None, "text_path is required for 'text' mode"
    elif opt.cond_mode == 'image+text':
        assert text_path is not None, "text_path is required for 'image+text' mode"
    elif opt.cond_mode == 'depth+image+text':
        assert text_path is not None and depth_path is not None, "text_path and depth_path are required for 'depth+image+text' mode"
    elif opt.cond_mode == 'image':
        assert text_path is None and depth_path is None, "text_path and depth_path should be None for 'image' mode"
    elif opt.cond_mode == 'image+depth':
        assert depth_path is not None, "depth_path is required for 'image+depth' mode"
    else:
        raise ValueError(f"Unsupported cond_mode: {opt.cond_mode}")

    def standard_image(rgb_path, target_height=512, target_width=512):
        image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0  # [H, W, 4]
        image = image[..., [2, 1, 0]]  # BGR to RGB
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # [C, H, W]
        height, width = image_tensor.shape[1], image_tensor.shape[2]

        if height > target_height:
            start_y = (height - target_height) // 2
            image_tensor = image_tensor[:, start_y:start_y + target_height, :]
        
        if width > target_width:
            start_x = (width - target_width) // 2
            image_tensor = image_tensor[:, :, start_x:start_x + target_width]

        if image_tensor.shape[1] < target_height or image_tensor.shape[2] < target_width:
            padded_image = torch.zeros((3, target_height, target_width), dtype=torch.float32)
            
            top_padding = (target_height - image_tensor.shape[1]) // 2
            bottom_padding = target_height - image_tensor.shape[1] - top_padding
            left_padding = (target_width - image_tensor.shape[2]) // 2
            right_padding = target_width - image_tensor.shape[2] - left_padding

            padded_image[:, top_padding:top_padding + image_tensor.shape[1], left_padding:left_padding + image_tensor.shape[2]] = image_tensor
            image_tensor = padded_image
        return image_tensor

    def standard_depth(depth_path, target_height=512, target_width=512):
        depth_image = np.load(depth_path).astype(np.float32)  # [H, W]
        depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).float()  # [0, 1]
        
        height, width = depth_tensor.shape[1], depth_tensor.shape[2]

        if height > target_height:
            start_y = (height - target_height) // 2
            depth_tensor = depth_tensor[:, start_y:start_y + target_height, :]

        if width > target_width:
            start_x = (width - target_width) // 2
            depth_tensor = depth_tensor[:, :, start_x:start_x + target_width]

        if depth_tensor.shape[1] < target_height or depth_tensor.shape[2] < target_width:
            padded_depth = torch.zeros((1, target_height, target_width), dtype=torch.float32)
            
            top_padding = (target_height - depth_tensor.shape[1]) // 2
            bottom_padding = target_height - depth_tensor.shape[1] - top_padding
            left_padding = (target_width - depth_tensor.shape[2]) // 2
            right_padding = target_width - depth_tensor.shape[2] - left_padding

            padded_depth[:, top_padding:top_padding + depth_tensor.shape[1], left_padding:left_padding + depth_tensor.shape[2]] = depth_tensor
            depth_tensor = padded_depth

        return depth_tensor

    if image_path is not None and opt.cond_mode != 'text':
        rgb = standard_image(image_path, target_height=opt.target_height, target_width=opt.target_width).to(device)
        rgb_show = rgb.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        rgb_batch = rgb.expand(1, -1, -1, -1)
        kiui.write_image(os.path.join(output_dir, f"{name}_rgb.png"), rgb_show)

    depth_batch = None
    if depth_path is not None:
        depth = standard_depth(depth_path, target_height=opt.target_height, target_width=opt.target_width)
        print(depth.shape)
        depth_show = depth.squeeze()
        plt.figure(figsize=(12, 12))
        sns.heatmap(depth_show, cmap='viridis')
        plt.savefig(os.path.join(output_dir, f"{name}_depth.png"))
        depth_batch = depth.expand(1, -1, -1, -1)

    if opt.cond_mode == 'text':
        conds = [text]
    elif opt.cond_mode == 'image':
        conds = rgb_batch
    elif opt.cond_mode == 'image+text':
        conds = [[text], rgb_batch]
    elif opt.cond_mode == 'image+depth':
        conds = [depth_batch, rgb_batch]
    elif opt.cond_mode == 'depth+image+text':
        conds = [[text], rgb_batch, depth_batch]
    else:
        raise ValueError(f"Unsupported cond_mode: {opt.cond_mode}")

    for i in range(opt.test_repeat):
        t0 = time.time()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                tokens = model.generate(conds, max_new_tokens=opt.test_max_seq_length, clean=True)
        t1 = time.time()
        print(f'[INFO] Processing, time = {t1 - t0:.4f}s')
        token = tokens[0]
        if token[:-1].shape[0] != opt.pose_length * 10:
            token = torch.tensor([256, 128, 128, 128, 128, 128, 128, 36, 64, 60]) / 256 * opt.discrete_bins
            token = token.repeat(opt.pose_length)
            coords = token.reshape(-1, 10)
        else:
            coords = token[:-1].reshape(-1, 10)
        coords = torch.tensor(coords, dtype=torch.float32)
        discrete_bins = opt.discrete_bins

        coords_traj = coords[:, :7]
        coords_instri = coords[:, 7:]
        coords_scale = coords_instri[:, -1]

        temp_traj = coords_traj / (0.5 * discrete_bins) - 1
        temp_instri = coords_instri / (discrete_bins / 10)
        scale = torch.exp(coords_scale / discrete_bins * 4 - 2)

        camera_tokens = torch.cat([temp_traj, temp_instri], dim=1)
        camera_tokens = camera_tokens.expand(1, -1, -1)
        camera_pose = token_to_camera(camera_tokens, 512, 512)
        
        c2ws = np.array(camera_pose[:, :, :12].cpu())
        scale_value = np.array(scale[0].cpu())
        c2ws = c2ws.reshape((-1, 3, 4))
        c2ws[:, :3, 3] = c2ws[:, :3, 3] * scale_value

        row_to_add = np.array([0, 0, 0, 1])
        c2ws = np.array([np.vstack((matrix, row_to_add)) for matrix in c2ws])
        
        def pose_normalize(camera_pose, pred_pose_path):
            camera_pose = camera_pose
            transforms_path = pred_pose_path

            f_x, f_y, c_x, c_y, w, h = camera_pose[0][0][-6:].tolist()
            # Create a dictionary of intrinsic parameters
            transforms_dict = {
                "w": w,
                "h": h,
                "fl_x": f_x,  # Focal length in x direction
                "fl_y": f_y,  # Focal length in y direction
                "cx": c_x,    # Principal point in x
                "cy": c_y,     # Principal point in y
                'frames': []
            }
            traj_tensor = camera_pose[:,:,:12]
            camera_list = []
            for i in range(120):
                t = torch.full((1, 1), fill_value=i/120)
                camera = sample_from_dense_cameras(traj_tensor, t)
                camera_list.append(camera[0])
            camera_tensor = torch.cat(camera_list, dim=0)  # Concatenate along the batch dimension (dim=0)
            camera_numpy = camera_tensor.clone().cpu().numpy()
            for idx, row in enumerate(camera_numpy):
                RT = row.reshape(3, 4)
                transform_matrix = np.vstack([RT, [0, 0, 0, 1]])
                transform_matrix_list = transform_matrix.tolist()
                frame_data = {
                    "transform_matrix": transform_matrix_list,
                    "monst3r_im_id": idx + 1  # Assuming colmap_im_id is an index starting from 1
                }
                transforms_dict['frames'].append(frame_data)

            with open(transforms_path, 'w') as f:
                json.dump(transforms_dict, f, indent=4)
                
        def save_results(output_dir, name, camera_pose):
            gt_caption_path = image_path.replace("_rgb.png", "_caption.json")
            new_caption_path = os.path.join(output_dir, f"{name}_caption.json")
            os.makedirs(os.path.dirname(new_caption_path), exist_ok=True)
            shutil.copy(gt_caption_path, new_caption_path)
            
            gt_pose_path = image_path.replace("_rgb.png", "_transforms_cleaning.json")
            new_pose_path = os.path.join(output_dir, f"{name}_transforms_ref.json")
            shutil.copy(gt_pose_path, new_pose_path)
            
            pred_pose_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
            pose_normalize(camera_pose, pred_pose_path)
        
        draw_json(c2ws, os.path.join(output_dir, f"{name}_traj.png"))
        gt_traj_path = image_path.replace("_rgb.png", "_traj_cleaning.png")
        new_traj_path = os.path.join(output_dir, f"{name}_traj_GT.png")
        shutil.copy(gt_traj_path, new_traj_path)
            
        save_results(output_dir, name, camera_pose)
            
        torch.cuda.synchronize()
        
assert opt.test_path is not None

output_dir = os.path.join(opt.workspace, opt.resume.split('/')[-1].split('.')[0])
if opt.cond_mode == 'text': 
    print("Start processing text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in sorted(image_paths):
            text_path = image_path.replace("_rgb.png", "_caption.json")
            name = 'test/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1][:-8]
            process_data(opt, output_dir, name, image_path, text_path)
elif opt.cond_mode == 'depth+image+text': 
    print("Start processing depth+image+text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in sorted(image_paths):
            text_path = image_path.replace("_rgb.png", "_caption.json")
            depth_path = image_path.replace("_rgb.png", "_depth.npy")
            name = 'test/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1][:-8]
            process_data(opt, output_dir, name, image_path, text_path, depth_path)
