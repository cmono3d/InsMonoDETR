import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import math

import yaml
import argparse
import datetime


from lib.helpers.dataloader_helper import build_dataloader

# from utils import box_ops
import cv2
import numpy as np


import sys
sys.path.append("..")
import os
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

        # {'Pedestrian': 3, 'Car': 1, 'Cyclist': 2}

obj_type = 'Car'

def read_detection(path,gt=True):
    global obj_type
    if os.path.getsize(path) == 0:
        return False


    df = pd.read_csv(path, header=None, sep=' ')

    # print(df)
    if gt:
        df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
    else:
        df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
        'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y','score']
    # df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
# #     df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    df = df[df['type']==obj_type]
    df.reset_index(drop=True, inplace=False)
    # print(df)
    return df


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

import cv2
import numpy as np
import logging
import torch



def create_logger(log_file, rank=0):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


root_dir = 'data/KITTIDataset/'

out_path = os.path.join(root_dir,'training/instance_2')
if not os.path.exists(out_path):
    os.makedirs(out_path)
log_file = os.path.join(out_path, '0missingoutput')
logger = create_logger(log_file)
####### 
with open(os.path.join(root_dir,'ImageSets/trainval.txt'),'r') as f:
    id_lst = f.readlines()
    id_lst = [id.strip() for id in id_lst]
# file_name_lst = sorted([img.split('.')[0] for img in img_lst])
for file_name in id_lst[1:]:
        
    image = cv2.imread(os.path.join(root_dir,f'training/image_2/{file_name}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    df = read_detection(os.path.join(root_dir,f'training/label_2/{file_name}.txt'))
    
    bboxes = np.array(df[['bbox_left', 'bbox_top','bbox_right', 'bbox_bottom']],dtype=np.int16)
    input_boxes = torch.tensor(bboxes).to(device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    try:

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        print(masks.shape)
        masks = masks.detach().cpu().numpy()
        h, w = masks.shape[-2:]
        new_mask = np.zeros(( h, w),dtype = np.uint16)
        # image = 
        # {'Pedestrian': 3, 'Car': 1, 'Cyclist': 2}
        for id in range(len(masks)):
            new_mask[masks[id].reshape(h, w)] = 1000+df.index[id]
        
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_path,f'{file_name}.png'),new_mask)
    except:
        logger.info(f'ground truth {file_name} has no {obj_type}')
            