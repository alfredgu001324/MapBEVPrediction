## Map Training & Evaluation

Whenever training and evaluating, please edit the config paths (and checkpoint paths if testing) in the bash file. Also, change the data paths in the config files to your nuscenes raw data and processed annotation data. 

During Evaluation, make sure to leave at least 1TB of storage space to store the BEV features. It would be great for subsequent work to apply this framework in an end-to-end manner to avoid storaging these BEV features. 

One potential way would be training online mapping as an auxilliary task. During inference, only utilize the BEV features for trajectory prediction in an end-to-end manner. 

### MapTR

**Training**

Run
```
cd MapTR_modified/
source train.sh      
```

Or by running:
```
export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTR_modified"

python tools/train.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  --deterministic \
  --no-validate
```

**Evaluation**

Run
```
cd MapTR_modified/
source test.sh                                  
```

Or by running:

```
export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTR_modified"

python tools/test.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  work_dirs/maptr_tiny_r50_24e/YOURCHECKPOINT.pth \
  --eval chamfer \
  --bev_path /path_to_save_bev_features
```


### MapTRv2

**Training**

Run
```
cd MapTRv2_modified/
source train.sh      
```

Or by running:
```
export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTRv2_modified"

python tools/train.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  --deterministic \
  --no-validate
```

**Evaluation**

Run
```
cd MapTRv2_modified/
source test.sh                                  
```

Or by running:

```
export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTRv2_modified"

python tools/test.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  work_dirs/maptrv2_nusc_r50_24ep/YOURCHECKPOINT.pth \
  --eval chamfer \
  --bev_path /path_to_save_bev_features
```

### StreamMapNet

**Training**

Run
```
cd StreamMapNet_modified/
source train.sh      
```

Or by running:
```
export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/StreamMapNet_modified"

python tools/train.py \
  plugin/configs/nusc_newsplit_480_60x30_24e.py \
  --deterministic \
  --no-validate
```

**Evaluation**

Run
```
cd StreamMapNet_modified/
source test.sh                                  
```

Or by running:

```
export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/StreamMapNet_modified"

python tools/test.py \
  plugin/configs/nusc_newsplit_480_60x30_24e.py \
  work_dirs/nusc_newsplit_480_60x30_24e/YOURCHECKPOINT.pth \
  --eval \
  --bev_path /path_to_save_bev_features
```