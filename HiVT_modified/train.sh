# if method is 'bev', choose [MapTR, MapTRv2, MapTRv2_CL, StreamMapNet]
python train.py \
  --root ../trj_data/{maptr,maptrv2,maptrv2_cent,stream} \
  --method {base, unc, bev} \
  --map_model MapTR \
  --embed_dim 128

