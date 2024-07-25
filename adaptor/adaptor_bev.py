import os
import pickle
import argparse
from tqdm import tqdm

def add_bev_features(scene_folder, bev_folder, map_model):
    scene_files = [f for f in os.listdir(scene_folder) if f.endswith('.pkl')]

    for scene_file in tqdm(scene_files, desc="Processing scenes"):
        scene_path = os.path.join(scene_folder, scene_file)

        with open(scene_path, 'rb') as f:
            scene_data = pickle.load(f)

        sample_idx = scene_data['sample_token']

        sample_file = f"{sample_idx}.pickle"
        sample_path = os.path.join(bev_folder, sample_file)

        with open(sample_path, 'rb') as f:
            bev_data = pickle.load(f)

        if isinstance(bev_data['bev'], torch.Tensor):
            bev_data['bev'] = bev_data['bev'].numpy()
            
        if map_model == 'MapTR':
            bev_features = bev_data['bev'].reshape(200, 100, 256)
        elif map_model == 'StreamMapNet':
            bev_features = np.flip(np.transpose(bev_data['bev'], (1, 2, 0)), axis=0).copy()

        scene_data['predicted_map']['bev_features'] = bev_features

        with open(scene_path, 'wb') as f:
            pickle.dump(scene_data, f)

    print("BEV features have been successfully added to the scene files.")

def main():
    parser = argparse.ArgumentParser(description="Add BEV features to scene files.")
    parser.add_argument('--scene_folder', type=str, required=True, help="Path to the folder containing scene pickle files")
    parser.add_argument('--bev_folder', type=str, required=True, help="Path to the folder containing bev features")
    parser.add_argument('--map_model', type=str, required=True, choices=['MapTR', 'StreamMapNet'])

    args = parser.parse_args()

    add_bev_features(args.scene_folder, args.sample_folder, args.map_model)

if __name__ == "__main__":
    main()
