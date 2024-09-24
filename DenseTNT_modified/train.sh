epochs=12
batch=1 
lr=0.00015
wd=0.05
dropout=0.2 
output_dir=/MapBEVPrediction/DenseTNT_modified/models/maptr_al
train_dir=/MapBEVPrediction/trj_data/maptr/train/data/
val_dir=/MapBEVPrediction/trj_data/maptr/val/data/

CUDA_LAUNCH_BLOCKING=1
python src/run.py \
  --method base_unc \
  --nuscenes \
  --argoverse \
  --argoverse2 \
  --future_frame_num 30 \
  --do_train \
  --data_dir $train_dir \
  --data_dir_for_val $val_dir \
  --output_dir $output_dir \
  --hidden_size 128 \
  --train_batch_size $batch \
  --use_map \
  --core_num 16 \
  --use_centerline \
  --distributed_training 0 \
  --other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph lane_scoring complete_traj complete_traj-3 \
  --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 \
  --learning_rate $lr \
  --weight_decay $wd \
  --hidden_dropout_prob $dropout \
  --num_train_epochs $epochs



