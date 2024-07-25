export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTRv2_modified"
python tools/train.py /MapBEVPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py --deterministic --no-validate
