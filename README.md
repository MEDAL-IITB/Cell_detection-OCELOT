# Cell Detection using Cell-Tissue Interaction - OCELOT Challenge 2023
https://ocelot2023.grand-challenge.org/

1. The ./cell_seg/ folder contains notebooks that are used to train segmentation models using only cell patches for cell detection
2. The ./celltissue/ folder contains notebook that are used to train segmentation model using both cell and tissue patches from the dataset
3. The ./classifier/ folder contains notebooks that are used to train a classifier to classify the detected cells as tumourous/background cells
4. The ./tissue_seg/ folder contains notebooks that are used to train a tissue segmentation model
5. The ./yolo_binary/ folder contains notebooks that are used to train a YOLOv8 model for cell detection
6. The ./ocelot23algo/ contains files used to build the docker image for submission and the evaluation code in ./ocelot23algo/evaluation/eval.py
