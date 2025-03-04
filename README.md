# MapBEVPrediction
This repository contains the official implementation of [Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](https://arxiv.org/abs/2407.06683) published in ECCV 2024. 

![](assets/flowchart.png)

## Getting Started
- [Environment Setup](docs/env.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Mapping Train and Eval](docs/map.md)
- [Merge Map and Trajectory Dataset](docs/adaptor.md)
- [Trajectory Train and Eval](docs/trj.md)
- [Visualization](docs/visualization.md)

## Results

Mapping checkpoints are [here](https://drive.google.com/drive/folders/1ZhCv4Z_yI7PLirHLS64luE1fV2EPG0sB?usp=sharing). Trajectory prediction checkpoints are [here](https://drive.google.com/drive/folders/1e0xRQqlIhYaAz3ErfNhZ57FCGN97vyzt?usp=sharing). 

## Dataset

I have uploaded all datasets (complete) for `MapTR`, `StreamMapNet`, `MapTRv2` and `MapTRv2 CL`. They are around 500GB each (StreamMapNet is around 200GB). You can download them through Hugging Face. They are located at [here](https://huggingface.co/datasets/alfredgu001324/MapBEVPrediction/tree/main) which should be easier to download.

Dataset Structure is as follows:
```
mapbevprediction
├── stream_bev/
├── maptr_bev/
├── maptrv2_bev/
│   ├── mini_val/
│   |   ├── data/
│   |   |   ├── scene-{scene_id}.pkl
│   ├── train/
│   ├── val/
├── maptrv2_cent_bev/
```

## Catalog

- [x] Visualization Code
- [x] Code release
  - [x] MapTR
  - [x] MapTRv2
  - [x] StreamMapNet
  - [x] HiVT
  - [x] DenseTNT
- [x] Untested version released + Instructions
- [x] Initialization

## Citation

If you found this repository useful, please consider citing our work:

```
@Inproceedings{GuSongEtAl2024,
  author    = {Gu, Xunjiang and Song, Guanyu and Gilitschenski, Igor and Pavone, Marco and Ivanovic, Boris},
  title     = {Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```

This codebase is built using our [prior work](https://github.com/alfredgu001324/MapUncertaintyPrediction), if your found this helpful, please also consider citing:

```
@Inproceedings{GuSongEtAl2024,
  author    = {Gu, Xunjiang and Song, Guanyu and Gilitschenski, Igor and Pavone, Marco and Ivanovic, Boris},
  title     = {Producing and Leveraging Online Map Uncertainty in Trajectory Prediction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

## License

This repository is licensed under [Apache 2.0](LICENSE).