## Visualization

The visualization bboxes are generated from [VAD](https://github.com/hustvl/VAD), thanks for their open-sourcing. You can download them from [here](https://drive.google.com/file/d/1f5SCMKJ6OkC-UuV_u2JXzWEDHYg94gQe/view?usp=drive_link). This is just for saving visualization time. 

Please run

```
cd MapTRv2_modified

python vis_bev.py \
  --version mini \                                   [mini, trainval]
  --dataroot ../nuscenes \
  --split mini_val \                                 [mini_val, val]
  --trj_pred HiVT \                                  [HiVT, DenseTNT]
  --map MapTR \                                      [MapTR, StreamMapNet]
  --trj_data ../trj_data/maptr/mini_val/data \
  --base_results ../PATH_to_baseline_prediction_results \
  --unc_results ../PATH_to_unc_prediction_results \
  --bev_results ../PATH_to_bev_prediction_results \
  --boxes bbox.pkl \
  --save_path $SAVE_PATH
```

Example demo



https://github.com/user-attachments/assets/bf2aff75-9e82-44e9-95d9-78dc7bf4eb22


