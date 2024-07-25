export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTR_modified"

python tools/test.py \
    /MapBEVPrediction/MapTR_modified/projects/configs/maptr/maptr_tiny_r50_24e.py \
    /MapBEVPrediction/MapTR_modified/work_dirs/maptr_tiny_r50_24e/YOURCHECKPOINT.pth \
    --eval chamfer \
    --bev_path /path_to_save_bev_features
