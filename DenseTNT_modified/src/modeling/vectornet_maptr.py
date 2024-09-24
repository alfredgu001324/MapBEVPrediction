from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from modeling.decoder import Decoder, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes
import utils
import matplotlib.pyplot as plt
from .vit_encoder import ViT
import math

class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def get_patch_idx(data, patch_height=1, patch_width=1, image_height=200, image_width=100):
    agent_pos = [np.array(data_['polygons']) for data_ in data]
    concatenated_array = np.concatenate(agent_pos, axis=0)
    rotated_coordinates = torch.tensor(concatenated_array).cuda()
 
    perception_height, perception_width = 60.0, 30.0

    scale_height = image_height / perception_height
    scale_width = image_width / perception_width

    scaled_coordinates = torch.clone(rotated_coordinates)
    scaled_coordinates[..., 0] *= scale_width
    scaled_coordinates[..., 1] *= scale_height

    translated_coordinates = torch.clone(scaled_coordinates)
    translated_coordinates[..., 0] += image_width / 2.0
    translated_coordinates[..., 1] = image_height / 2.0 - translated_coordinates[..., 1]

    pixel_coordinates = translated_coordinates

    patch_indices = pixel_coordinates // torch.tensor([patch_width, patch_height]).to(rotated_coordinates.device)
    patch_indices = patch_indices[..., [1, 0]]

    v_patch, h_patch = image_height/patch_height, image_width/patch_width 

    patch_indices[..., 0] = torch.clamp(patch_indices[..., 0], min=0, max=int(v_patch)-1)
    patch_indices[..., 1] = torch.clamp(patch_indices[..., 1], min=0, max=int(h_patch)-1)

    return patch_indices[:, :, :]


class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(NewSubGraph, self).__init__()
        if depth is None:
            depth = args.sub_graph_depth

        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])
        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layer_0_again = MLP(hidden_size)


    def forward(self, input_list: list, bev=None, map=False):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = utils.merge_tensors(input_list, device)

        if bev is not None:
            hidden_states = torch.cat([hidden_states, bev], dim=-1)

        hidden_size = hidden_states.shape[2]
        max_vector_num = hidden_states.shape[1]
        attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states)                                   # num_polylines_per_batch, 20, 128

        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)


        return torch.max(hidden_states, dim=1)[0], torch.cat(utils.de_merge_tensors(hidden_states, lengths))


class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self, args_: utils.Args):
        super(VectorNet, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size

        self.point_level_sub_graph = NewSubGraph(hidden_size)
        self.point_level_sub_graph_map = NewSubGraph(2*hidden_size)
        self.point_level_cross_attention = CrossAttention(hidden_size)

        self.global_graph = GlobalGraph(hidden_size)

        # Use multi-head attention and residual connection.
        if 'enhance_global_graph' in args.other_params:
            self.global_graph = GlobalGraphRes(hidden_size)
        if 'laneGCN' in args.other_params:
            self.laneGCN_A2L = CrossAttention(hidden_size)
            self.laneGCN_L2L = GlobalGraphRes(hidden_size)
            self.laneGCN_L2A = CrossAttention(hidden_size)

        self.decoder = Decoder(args, self)

        if 'complete_traj' in args.other_params:
            self.decoder.complete_traj_cross_attention = CrossAttention(hidden_size)
            self.decoder.complete_traj_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=self.decoder.future_frame_num * 2)
        
        self.processing_layer = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),  
            nn.Dropout(0.3)
        )

        self.input_proj = nn.Conv2d(
            2*hidden_size, hidden_size, kernel_size=1)
        
        self.bev_pos_embed = SinePositionalEncoding(num_feats=hidden_size//2, normalize=True)

    def _prepare_context(self, bev_features):
        """Prepare class label and vertex context."""
        device = bev_features.device

        # Add 2D coordinate grid embedding
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W)
 
        bev_pos_embeddings = self.bev_pos_embed(bev_mask) # (bs, embed_dims, H, W)
        bev_features = self.input_proj(bev_features) # + bev_pos_embeddings # (bs, embed_dims, H, W)
    
        return bev_features.permute(0, 2, 3, 1)

    def forward_encode_sub_graph(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[List[slice]],
                                 device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        patch_indices = get_patch_idx(mapping)                                      # 316, 20, 2
        num_polygons = [len(item['polygons']) for item in mapping]
        new_patch_indices = (patch_indices[:, :, 0] * 100 + patch_indices[:, :, 1]).type(torch.long)

        bev_embed = torch.stack([data_['bev_embed'] for data_ in mapping]).cuda()
        bev_embed = self._prepare_context(bev_embed.permute(0, 3, 1, 2))
        replicated_bev_embed = torch.cat([bev_embed[i].repeat(count, 1, 1, 1) for i, count in enumerate(num_polygons)])
        replicated_bev_embed = replicated_bev_embed.reshape(sum(num_polygons), -1, bev_embed.shape[-1])

        polygon_indices = torch.arange(0, replicated_bev_embed.shape[0]).unsqueeze(-1) 
        polygon_indices = polygon_indices.expand(-1, new_patch_indices.shape[1])
        selected_features = replicated_bev_embed[polygon_indices, new_patch_indices, :]         # num_polylines, 20, 256

        batch_features = torch.split(selected_features, num_polygons, dim=0)

        input_list_list = []

        map_input_list_list = []
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []
            map_input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span], device=device)
                input_list.append(tensor)
                if j >= map_start_polyline_idx:
                    map_input_list.append(tensor)

            input_list_list.append(input_list)    
            map_input_list_list.append(map_input_list)

        if True:
            element_states_batch = []
            for i in range(batch_size):
                a, b = self.point_level_sub_graph(input_list_list[i])
                element_states_batch.append(a)

        if 'lane_scoring' in args.other_params:
            lane_states_batch = []
            for i in range(batch_size):
                a, b = self.point_level_sub_graph_map(map_input_list_list[i], bev=batch_features[i], map=True)
                a = self.processing_layer(a)
                lane_states_batch.append(a)

        # We follow laneGCN to fuse realtime traffic information from agent nodes to lane nodes.
        if 'laneGCN' in args.other_params:
            for i in range(batch_size):
                map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
                agents = element_states_batch[i][:map_start_polyline_idx]
                lanes = element_states_batch[i][map_start_polyline_idx:]
                # Origin laneGCN contains three fusion layers. Here one fusion layer is enough.
                if True:
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)
                else:
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), agents.unsqueeze(0)).squeeze(0)
                    lanes = lanes + self.laneGCN_L2L(lanes.unsqueeze(0)).squeeze(0)
                    agents = agents + self.laneGCN_L2A(agents.unsqueeze(0), lanes.unsqueeze(0)).squeeze(0)
                
                element_states_batch[i] = torch.cat([agents, lanes])

        return element_states_batch, lane_states_batch

    # @profile
    def forward(self, mapping: List[Dict], device):
        import time
        global starttime
        starttime = time.time()

        matrix = utils.get_from_mapping(mapping, 'matrix')
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')

        batch_size = len(matrix)

        if args.argoverse:
            utils.batch_init(mapping)

        element_states_batch, lane_states_batch = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)

        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)

        utils.logging('time3', round(time.time() - starttime, 2), 'secs')

        return self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device)
