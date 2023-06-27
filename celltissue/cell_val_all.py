import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from tqdm import tqdm
import cv2
from skimage import feature
import json
import os

#cv2.setNumThreads(1)

val = glob.glob('/workspace/jay/DDP/Ocelot/yolo_binary/datasets/cell_detect_33-1/valid/images/*.jpg')
test = glob.glob('/workspace/jay/DDP/Ocelot/yolo_binary/datasets/cell_detect_33-1/test/images/*.jpg')
train = sorted(glob.glob('/workspace/jay/DDP/Ocelot/ocelot2023/images/train/cell/*.jpg'))
val_files = np.unique(np.array([x.split('/')[-1][:3] for x in val]))
test_files = np.unique(np.array([x.split('/')[-1][:3] for x in test]))
train_files = np.unique(np.array([x.split('/')[-1][:3] for x in train]))

device = 'cuda:2'
tissue_seg_model = torch.load('/workspace/jay/DDP/Ocelot/tissue_seg/sub_ckpts/41_0.080.pt',map_location=device)
tissue_seg_model = tissue_seg_model.eval()
softmax = torch.nn.Softmax(dim=1)

with open('/workspace/jay/DDP/Ocelot/ocelot2023/metadata.json') as f:
    jsonn = json.load(f)
    
def find_cells(heatmap,min_dist=8):
    """This function detects the cells in the output heatmap
    Parameters
    ----------
    heatmap: torch.tensor
        output heatmap of the model,  shape: [1, 3, 1024, 1024]
    Returns
    -------
        List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    """
    arr = heatmap[0,:,:,:].cpu().detach().numpy()
    # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

    pred_wo_bg,bg = np.split(arr, (2,), axis=0) # Background and non-background channels
    bg = np.squeeze(bg, axis=0)
    obj = 1.0 - bg

    arr = cv2.GaussianBlur(obj, (5,5), sigmaX=3)
    peaks = feature.peak_local_max(
        arr, min_distance=min_dist, exclude_border=0, threshold_abs=0.0
    ) # List[y, x]

    maxval = np.max(pred_wo_bg, axis=0)
    maxcls_0 = np.argmax(pred_wo_bg, axis=0)

    # Filter out peaks if background score dominates
    peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
    if len(peaks) == 0:
        return []

    # Get score and class of the peaks
    scores = maxval[peaks[:, 0], peaks[:, 1]]
    peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

    predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

    return predicted_cells

#eval_files = [val_files,test_files]
#model_paths = sorted(glob.glob('/workspace/jay/DDP/Ocelot/celltissue/deeplab_dice_ckpts_v2/*.pt'))

for k in range(200):
    print(f"Epoch: {k}")
    path = glob.glob(f'/workspace/jay/DDP/Ocelot/celltissue/deeplab_dice_ckpts_v2/{k}_*.pt')
    model = torch.load(path[0],map_location=device)
    model = model.eval()
    pred_json = {
    "type": "Multiple points",
    "num_images": len(train_files),
    "points": [],
    "version": {
        "major": 1,
        "minor": 0,
        }
    }
    for j,file in enumerate((val_files)):
        idx = int(file) -1 
        cell_path = f'/workspace/jay/DDP/Ocelot/ocelot2023/images/train/cell/{file}.jpg'
        tissue_path = f'/workspace/jay/DDP/Ocelot/ocelot2023/images/train/tissue/{file}.jpg'
        cell = (np.array(Image.open(cell_path))/255) - 0.5
        cell = torch.Tensor(np.moveaxis(cell, -1, 0))
        cell = cell[None,:]
        cell = cell.to(device)
        tissue = (np.array(Image.open(tissue_path))/255) - 0.5
        tissue = torch.Tensor(np.moveaxis(tissue, -1, 0))
        tissue = tissue[None,:]
        tissue = tissue.to(device)   
        with torch.no_grad():
            tissue_out = softmax(tissue_seg_model(tissue))
        image = torch.zeros((1,6,1024,1024)).to(device)
        for i in range(1):
            yc = int(jsonn['sample_pairs'][file]['patch_x_offset']*1024)
            xc = int(jsonn['sample_pairs'][file]['patch_y_offset']*1024)
            tissue_crop = np.moveaxis(tissue_out[i].cpu().numpy(),0,-1)
            tissue_crop = tissue_crop[xc-128:xc+128,yc-128:yc+128,:]
            tissue_crop = cv2.resize(tissue_crop, dsize=(1024,1024), interpolation = cv2.INTER_NEAREST)   
            tissue_crop = torch.Tensor(np.moveaxis(tissue_crop,-1,0))
            image[i] = torch.concat((cell[i],tissue_crop.to(device)),dim=0)
        with torch.no_grad():
            out_mask = softmax(model(image))
        predicted_cells = find_cells(out_mask,min_dist=10)
        for i in range(len(predicted_cells)):
            x,y,clas,prob = predicted_cells[i]
            point = {
                    "name": f"image_{idx}",
                    "point": [int(x), int(y), int(clas)],
                    "probability": prob,  # dummy value, since it is a GT, not a prediction
                    }
            pred_json["points"].append(point)
    del model
    with open("/workspace/jay/DDP/Ocelot/ocelot23algo/evaluation/celltissueseg_val.json", "w") as g:
        json.dump(pred_json, g)
        print("JSON file saved")
    os.system('python /workspace/jay/DDP/Ocelot/ocelot23algo/evaluation/eval.py')