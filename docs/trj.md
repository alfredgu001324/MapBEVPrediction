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