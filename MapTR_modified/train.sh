export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTR_modified"
python tools/train.py /MapBEVPrediction/MapTR_modified/projects/configs/maptr/maptr_tiny_r50_24e.py --deterministic --no-validate
