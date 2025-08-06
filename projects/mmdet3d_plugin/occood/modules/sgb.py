import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
import numpy as np
from .vrwkv import Block as RWKVBlock
class BasicBlock(spconv.SparseModule):
    def __init__(self, C_in, C_out, indice_key):
        super(BasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(C_out, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out)
        )
        self.relu2 = spconv.SparseSequential(
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        identity = self.layers_in(x)
        out = self.layers(x)
        output = spconv.SparseConvTensor(sum([i.features for i in [identity, out]]),
                                         out.indices, out.spatial_shape, out.batch_size)
        output.indice_dict = out.indice_dict
        output.grid = out.grid
        return self.relu2(output)


def make_layers_sp(C_in, C_out, blocks, indice_key):
    layers = []
    layers.append(BasicBlock(C_in, C_out, indice_key))
    for _ in range(1, blocks):
        layers.append(BasicBlock(C_out, C_out, indice_key))
    return spconv.SparseSequential(*layers)


def scatter(x, idx, method, dim=0):
    if method == "max":
        return torch_scatter.scatter_max(x, idx, dim=dim)[0]
    elif method == "mean":
        return torch_scatter.scatter_mean(x, idx, dim=dim)
    elif method == "sum":
        return torch_scatter.scatter_add(x, idx, dim=dim)
    else:
        print("unknown method")
        exit(-1)


class SFE(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, layer_name, layer_num=1):
        super().__init__()
        self.spconv_layers = make_layers_sp(in_channels, out_channels, layer_num, layer_name)

    def forward(self, inputs):
        conv_features = self.spconv_layers(inputs)
        return conv_features


class SGFE(nn.Module):
    def __init__(self, input_channels, output_channels, reduce_channels, name, p_scale=[2, 4, 6, 8]):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name

        self.feature_reduce = nn.Linear(input_channels, reduce_channels)
        self.pooling_scale = p_scale
        self.fc_list = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _, _ in enumerate(self.pooling_scale):
            self.fc_list.append(nn.Sequential(
            nn.Linear(reduce_channels, reduce_channels//2),
            nn.ReLU(),
            ))
            self.fcs.append(nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2)))
        self.scale_selection = nn.Sequential(
            nn.Linear(len(self.pooling_scale) * reduce_channels//2,
                                       reduce_channels),nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2, bias=False),
                                nn.ReLU(inplace=False))
        self.out_fc = nn.Linear(reduce_channels//2, reduce_channels, bias=False)
        self.linear_output = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduce_channels, output_channels),
        )

    def forward(self, input_data, input_coords):
        reduced_feature = F.relu(self.feature_reduce(input_data))
        output_list = [reduced_feature]
        for j, ps in enumerate(self.pooling_scale):
            index = torch.cat([input_coords[:, 0].unsqueeze(-1),
                              (input_coords[:, 1:] // ps).int()], dim=1)
            _, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            fkm = scatter(reduced_feature, unq_inv, method="mean", dim=0)
            att = self.fc_list[j](fkm)[unq_inv]
            output_list.append(att)
        scale_features = torch.stack(output_list[1:], dim=1)
        feat_S = scale_features.sum(1)
        feat_Z = self.fc(feat_S)
        attention_vectors = [fc(feat_Z) for fc in self.fcs]
        attention_vectors = torch.sigmoid(torch.stack(attention_vectors, dim=1))
        scale_features = self.out_fc(torch.sum(scale_features * attention_vectors, dim=1))

        output_f = torch.cat([reduced_feature, scale_features], dim=1)
        proj = self.linear_output(output_f)
        
        return proj


class Sem_RWKV(nn.Module):
    def __init__(self, sizes=[128, 128, 16], channels=128):
        super().__init__()
        self.sizes = sizes

        c = channels
        self.conv_block = SFE(c, c, "svpfe")
        self.proj_block = SGFE(input_channels=c, output_channels=c, reduce_channels=c, name="proj")

        self.conv_block_down = SFE(c, c, "svpfe_down")
        self.proj_block_down = SGFE(input_channels=c, output_channels=c, reduce_channels=c, name="proj_down")

        self.conv_block_down_1 = SFE(c, c, "svpfe_down_1")
        self.proj_block_down_1 = SGFE(input_channels=c, output_channels=c, reduce_channels=c, name="proj_down_1")

        self.rwkv_block1 = RWKVBlock(n_embd=c, n_layer=18, layer_id=0)
        self.rwkv_block2 = RWKVBlock(n_embd=c, n_layer=18, layer_id=0)
        self.rwkv_block3 = RWKVBlock(n_embd=c, n_layer=18, layer_id=0)

        self.fc_out = nn.Linear(3*c, c)
        self.out2 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 20)
        )
        self.out4 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 20)
        )
        self.out8 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 20)
        )
    def bev_projection(self, vw_features, vw_coord, sizes, batch_size):
        unq, unq_inv = torch.unique(
            torch.cat([vw_coord[:, 0].unsqueeze(-1), vw_coord[:, -2:]], dim=-1).int(), return_inverse=True, dim=0
        )
        bev_fea = scatter(vw_features, unq_inv, method='max')
        bev_dense = spconv.SparseConvTensor(bev_fea, unq.int(), sizes[-2:], batch_size).dense()  # B, C, H, W
        return bev_dense
    
    def forward(self, vw_features, coord_ind):
        #1，128，128，128，16
        batch_size = torch.unique(coord_ind[:,0]).max() + 1
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)

        input_tensor = spconv.SparseConvTensor(
            vw_features, coord.int(), np.array(self.sizes, np.int32)[::-1], batch_size
        )
        
        conv_output = self.conv_block(input_tensor)
        proj_vw1 = self.proj_block(conv_output.features, input_coords=coord.int())

       
        bev1 = self.bev_projection(proj_vw1, coord, np.array(self.sizes, np.int32)[::-1], batch_size)
        B, C, H, W = bev1.shape
        bev1_rwkv = self.rwkv_block1(bev1.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=(H, W))
        bev1_rwkv = bev1_rwkv.view(B, H, W, C).permute(0, 3, 1, 2)#b,128,128,128

        # downsample
        index = torch.cat([coord[:, 0].unsqueeze(-1),
                          (coord[:, 1:] // 2).int()], dim=1)
        unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
        proj_vw1_down = scatter(proj_vw1, unq_inv, method="max", dim=0)

        input_tensor_down = spconv.SparseConvTensor(
            proj_vw1_down, unq.int(), (np.array(self.sizes, np.int32)//2)[::-1], batch_size
        )
        
        conv_output_down = self.conv_block_down(input_tensor_down)
        proj_vw2 = self.proj_block_down(conv_output_down.features, input_coords=unq.int())
        
        bev2 = self.bev_projection(proj_vw2, unq, (np.array(self.sizes, np.int32) // 2)[::-1], batch_size)
        B, C, H, W = bev2.shape
        bev2_rwkv = self.rwkv_block2(bev2.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=(H, W))
        bev2_rwkv = bev2_rwkv.view(B, H, W, C).permute(0, 3, 1, 2)
        #1，128，64，64，4
        # downsample
        index_1 = torch.cat([unq[:, 0].unsqueeze(-1),
                          (unq[:, 1:] // 4).int()], dim=1)
        unq_1, unq_inv_1 = torch.unique(index_1, return_inverse=True, dim=0)
        proj_vw2_down = scatter(proj_vw2, unq_inv_1, method="max", dim=0)


        input_tensor_down_1 = spconv.SparseConvTensor(
            proj_vw2_down, unq_1.int(), (np.array(self.sizes, np.int32)//4)[::-1], batch_size
        )
        conv_output_down_1 = self.conv_block_down_1(input_tensor_down_1)
        proj_vw3 = self.proj_block_down_1(conv_output_down_1.features, input_coords=unq_1.int())
        
        bev3 = self.bev_projection(proj_vw3, unq_1, (np.array(self.sizes, np.int32) // 4)[::-1], batch_size)
        B, C, H, W = bev3.shape
        bev3_rwkv = self.rwkv_block3(bev3.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=(H, W))
        bev3_rwkv = bev3_rwkv.view(B, H, W, C).permute(0, 3, 1, 2)

        proj_vw = self.fc_out(torch.cat([vw_features, proj_vw1, proj_vw2[unq_inv]], dim=-1))

        sem_2 = self.out2(proj_vw1)
        sem_4 = self.out4(proj_vw2)
        sem_8 = self.out8(proj_vw3)

        return proj_vw,dict(
            mss_bev_dense = [bev1_rwkv, bev2_rwkv, bev3_rwkv],
            mss_logits_list = [sem_2, sem_4, sem_8],
            coord_list = [coord, unq, unq_1]
            )
