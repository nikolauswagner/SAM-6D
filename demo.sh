# set the paths
export CAD_PATH=../Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=../Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=../Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=../Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=../Data/Example/outputs         # path to a pre-defined file for saving results

# Render CAD templates
cd Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 


# Run instance segmentation model
export SEGMENTOR_MODEL=fastsam

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH


# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

