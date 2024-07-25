# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV1DataModule
from datamodules import NuscenesDataModule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--method', type=str, default='base', choices=['base', 'unc', 'bev'])
    parser.add_argument('--map_model', type=str, choices=['MapTR', 'MapTRv2', 'MapTRv2_CL', 'StreamMapNet'], help='Specify the map model.')
    parser.add_argument('--centerline', action='store_true', help='centerline usage')

    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.method == 'bev' and args.map_model is None:
        parser.error('--map_model is required when bev is chosen.')

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])
    model = HiVT(**vars(args))
    model.method = args.method
    model.map_model = args.map_model

    datamodule = NuscenesDataModule.from_argparse_args(args)
    trainer.fit(model, datamodule)
