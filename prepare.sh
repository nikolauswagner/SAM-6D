#! /bin/bash

### Create conda environment
conda env create -f environment.yaml
conda activate sam6d

### Install pointnet2
cd Pose_Estimation_Model/model/pointnet2
python3 setup.py install
cd ../../../

### Download ISM pretrained model
cd Instance_Segmentation_Model
python3 download_sam.py
python3 download_fastsam.py
python3 download_dinov2.py
cd ../

### Download PEM pretrained model
cd Pose_Estimation_Model
python3 download_sam6d-pem.py
