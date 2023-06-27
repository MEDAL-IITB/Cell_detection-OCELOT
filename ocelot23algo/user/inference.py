import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from skimage import feature

cv2.setNumThreads(1)

class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata
        self.celltissue_model = torch.load('/opt/app/celltissue.pt',map_location=torch.device('cpu'))
        self.celltissue_model = self.celltissue_model.eval()
        
        self.tissue_seg_model = torch.load('/opt/app/tissue_seg.pt',map_location=torch.device('cpu'))
        self.tissue_seg_model = self.tissue_seg_model.eval()
        
        self.softmax = torch.nn.Softmax(dim=1)
    
    def find_cells(self,heatmap,min_dist=10):
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

    def __call__(self, cell_patch, tissue, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided. 

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
        
        yc = int(meta_pair['patch_x_offset']*1024)
        xc = int(meta_pair['patch_y_offset']*1024)
                
        xs = []
        ys = []
        class_id = []
        probs = []
        
        cell = (cell_patch/255) - 0.5
        cell = torch.Tensor(np.moveaxis(cell, -1, 0))
        cell = cell[None,:]
        tissue = (tissue/255) - 0.5
        tissue = torch.Tensor(np.moveaxis(tissue, -1, 0))
        tissue = tissue[None,:]  
        with torch.no_grad():
            tissue_out = self.softmax(self.tissue_seg_model(tissue))
        image = torch.zeros((1,6,1024,1024))
        for i in range(1):
            tissue_crop = np.moveaxis(tissue_out[i].cpu().numpy(),0,-1)
            tissue_crop = tissue_crop[xc-128:xc+128,yc-128:yc+128,:]
            tissue_crop = cv2.resize(tissue_crop, dsize=(1024,1024), interpolation = cv2.INTER_NEAREST)   
            tissue_crop = torch.Tensor(np.moveaxis(tissue_crop,-1,0))
            image[i] = torch.concat((cell[i],tissue_crop),dim=0)
        with torch.no_grad():
            out_mask = self.softmax(self.celltissue_model(image))
        predicted_cells = self.find_cells(out_mask,min_dist=10)
        for i in range(len(predicted_cells)):
            x,y,clas,prob = predicted_cells[i]
            xs.append(int(x))
            ys.append(int(y))
            class_id.append(int(clas))
            probs.append(prob) 
                
        
        # The following is a dummy cell detection algorithm
#         prediction = np.copy(cell_patch[:, :, 2])
#         prediction[(cell_patch[:, :, 2] <= 40)] = 1
#         xs, ys = np.where(prediction.transpose() == 1)
#         class_id = [1] * len(xs) # Type of cell
#         probs = [1.0] * len(xs) # Confidence score

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, class_id, probs))
