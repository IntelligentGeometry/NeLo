
import torch
import torch.nn
from typing import Dict, Optional

import torch_geometric

from src.graphtree.graph_tree import GraphTree

from src.modules.resblocks import *

from src.modules.blocks import *

from config.global_config import global_config


class GraphUNet(torch.nn.Module):
    r''' UNet but with graph neural network
    no octree data structure
  
    use graphtree as the substitution of octree
    '''

    def __init__(self, in_channels: int, out_channels: int, interp: str = 'linear',
                 nempty: bool = False, **kwargs):
        super(GraphUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)

        # encoder
        self.conv1 = GraphConvBnRelu(in_channels, self.encoder_channel[0])

        self.downsample = torch.nn.ModuleList(
            [PoolingGraph() for i in range(self.encoder_stages)]
        )
        self.encoder = torch.nn.ModuleList(
            [GraphResBlocks(self.encoder_channel[i], self.encoder_channel[i + 1],
                            resblk_num=self.encoder_blocks[i], resblk=self.resblk, bottleneck=self.bottleneck)
             for i in range(self.encoder_stages)]
        )

        # decoder
        channel = [self.decoder_channel[i] + self.encoder_channel[-i - 2]
                   for i in range(self.decoder_stages)]
        self.upsample = torch.nn.ModuleList(
            [UnpoolingGraph() for i in range(self.decoder_stages)]
        )
        self.decoder = torch.nn.ModuleList(
            [GraphResBlocks(channel[i], self.decoder_channel[i + 1],
                            resblk_num=self.decoder_blocks[i], resblk=self.resblk, bottleneck=self.bottleneck)
             for i in range(self.decoder_stages)]
        )

        # header
        self.header = torch.nn.Sequential(
            Conv1x1BnRelu(self.decoder_channel[-1], self.decoder_channel[-1]),
            Conv1x1(self.decoder_channel[-1], self.out_channels, use_bias=True))
        
     


    def config_network(self):
        r''' Configure the network channels and Resblock numbers.
        '''
        # self.encoder_blocks = [2, 2, 2, 3]
        # self.decoder_blocks = [2, 2, 2, 3]
        # self.encoder_channel = [32, 32, 32, 32, 64]
        # self.decoder_channel = [64, 32, 32, 32, 32]

        self.encoder_blocks = global_config.unet_encoder_blocks
        self.decoder_blocks = global_config.unet_decoder_blocks
        self.encoder_channel = global_config.unet_encoder_feature_dim
        self.decoder_channel = global_config.unet_decoder_feature_dim

        self.bottleneck = 1
        self.resblk = GraphResBlock2

    def unet_encoder(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
        r''' The encoder of the U-Net.
        '''

        convd = dict()
        convd[depth] = self.conv1(data, graphtree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], graphtree, i + 1)
            convd[d - 1] = self.encoder[i](conv, graphtree, d - 1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], graphtree: GraphTree, depth: int):
        r''' The decoder of the U-Net. 
        '''

        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, graphtree, self.decoder_stages - i)
            deconv = torch.cat([convd[d + 1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, graphtree, d + 1)
        return deconv

    def forward(self, data: torch.Tensor, graphtree: GraphTree, depth: int,
                return_graph_level_feature=False,
                base_k=1,  
                ):
        """_summary_
    
        Args:
            data (torch.Tensor): _description_
            graphtree (GraphTree): _description_
            depth (int): _description_
        """

        convd = self.unet_encoder(data, graphtree, depth)
        deconv = self.unet_decoder(convd, graphtree, depth - self.encoder_stages)

        # interp_depth = depth - self.encoder_stages + self.decoder_stages
        # feature = self.octree_interp(deconv, graphtree, interp_depth, query_pts)
        final = self.header(deconv)

        if return_graph_level_feature == True:
            graphwise_embedding = torch_geometric.nn.global_mean_pool(deconv, batch=graphtree.treedict[0].batch)
            return final, graphwise_embedding

        return final


