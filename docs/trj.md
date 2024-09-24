### HiVT 

#### Setup

Please follow [HiVT](https://github.com/ZikangZhou/HiVT) setup guide to set up environment.

**Note** that different maps have different encoder configurations for optimal perforamance

#### Training

```
cd HiVT_modified

# if method is 'bev', choose [MapTR, MapTRv2, MapTRv2_CL, StreamMapNet]
python train.py \
  --root ../trj_data/{maptr,maptrv2,maptrv2_cent,stream} \
  --method {base, unc, bev} \
  --map_model {MapTR, MapTRv2, MapTRv2_CL, StreamMapNet} \
  --embed_dim 128
```

For training MapTRv2 Centerline, add an `--centerline` argument. 

#### Testing

```
cd HiVT_modified

# if method is 'bev', choose [MapTR, MapTRv2, MapTRv2_CL, StreamMapNet]
python eval.py \
  --root ../trj_data/{maptr, maptrv2, maptrv2_cent, stream} \
  --split {mini_val, val} \
  --method {base, unc, bev} \
  --map_model {MapTR, MapTRv2, MapTRv2_CL, StreamMapNet} \
  --batch_size 32 \
  --ckpt_path /path/to/your_checkpoint.ckpt
```

For evaluating MapTRv2 Centerline, add an `--centerline` argument. 

#### Visualization

Please uncomment [this line](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/8ab64116982303d373eb85fea2501e139a09e781/HiVT_modified/models/hivt.py#L138) to save the pkl files necessary for [later visualization](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/docs/visualization.md).

### DenseTNT

#### Setup

Please follow [DenseTNT](https://github.com/Tsinghua-MARS-Lab/DenseTNT/tree/main) setup guide to set up environment.


#### Training

You can use the `src/train.sh` or by running the following:
```
cd DenseTNT_modified

# Please adjust the hyperparemters based on the paper's Appendix
epochs=12
batch=1 
lr=0.00015
wd=0.05
dropout=0.2 
output_dir=/MapBEVPrediction/DenseTNT_modified/models/maptr_al
train_dir=/MapBEVPrediction/trj_data/maptr/train/data/
val_dir=/MapBEVPrediction/trj_data/maptr/val/data/

python src/run.py \
  --method {base_unc, maptr_bev, stream_bev} \
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
```

#### Testing

You can use `src/eval.sh` or by running the following:

```
cd DenseTNT_modified

# Please adjust the hyperparemters based on the paper's Appendix
epochs=12
batch=1 
lr=0.00015
wd=0.05
dropout=0.2 
output_dir=/MapBEVPrediction/DenseTNT_modified/models/maptr_al
train_dir=/MapBEVPrediction/trj_data/maptr/train/data/
val_dir=/MapBEVPrediction/trj_data/maptr/val/data/

CUDA_LAUNCH_BLOCKING=1
i=12    # Or any checkpoint number you want to evaluate

python src/run.py \
  --method base_unc \
  --nuscenes \
  --argoverse \
  --argoverse2 \
  --future_frame_num 30 \
  --do_eval \
  --data_dir $train_dir \
  --data_dir_for_val $val_dir \
  --output_dir $output_dir \
  --hidden_size 128 \
  --train_batch_size $batch \
  --eval_batch_size 16 \
  --use_map \
  --core_num 16 \
  --use_centerline \
  --distributed_training 0 \
  --other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph lane_scoring complete_traj complete_traj-3 \
  --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 \
  --learning_rate $lr \
  --weight_decay $wd \
  --hidden_dropout_prob $dropout \
  --model_recover_path $i
```