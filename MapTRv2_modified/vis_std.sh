dataset='mini' #mini, full
map_model='maptr' #stream
pred_model='HiVT_modified' # DenseTNT_modified
processed_data="/home/guxunjia/project/HiVT_data/maptr/${dataset}_val/data/"
predict_data="/home/guxunjia/project/${pred_model}/bev_results/${dataset}/${map_model}_al.pkl"
predict_data_unc="/home/guxunjia/project/${pred_model}/bev_results/${dataset}/${map_model}_al_unc.pkl"
# camera_path="/home/guxunjia/project/MapTR/work_dirs/maptr_tiny_r50_24e/vis_pred_mini_val/"
# save_path="/home/guxunjia/Desktop/VAD/CVPR_gifs/HiVT_MapTR/try_2/" # /home/guxunjia/Desktop/VAD/CVPR_gifs/HiVT_MapTR/mp4/
# save_path="/home/guxunjia/Desktop/VAD/CVPR_gifs/full/HiVT_MapTR/"
save_path="/home/guxunjia/MapUncertaintyPrediction/test/"

python vis_std.py \
  --version mini \
  --dataroot /home/data/nuscenes \
  --split mini_val \
  --trj_pred HiVT \
  --map MapTR \
  --trj_data $processed_data \
  --base_results $predict_data \
  --unc_results $predict_data_unc \
  --boxes /home/guxunjia/Desktop/VAD/test/full_bbox.pkl \
  --save_path $save_path 