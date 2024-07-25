export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTRv2_modified"
python tools/test.py \
    /MapBEVPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
    /MapBEVPrediction/MapTRv2_modified/work_dirs/maptrv2_nusc_r50_24ep/YOURCHECKPOINT.pth \
    --eval chamfer \
    --bev_path /path_to_save_bev_features

# python tools/test.py \
#     /MapBEVPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py \
#     /MapBEVPrediction/MapTRv2_modified/work_dirs/maptrv2_nusc_r50_24ep_w_centerline/YOUR_CHECKPOINT.pth \
#     --eval chamfer \
#     --bev_path /path_to_save_bev_features
