
import torch
import random
import shutil
import math
from collections import OrderedDict, defaultdict
import csv
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from mamba_ssm import Mamba2


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph( A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph( num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

class unit_gcn(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim,
        A
    ):
        super().__init__()
        self.dim = dim
        self.norm_ov = nn.BatchNorm2d(dim)
        self.num_subsets = A.size(0)
        #A = torch.from_numpy(A).float()
        self.mid_channels = dim // 8
        
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets), requires_grad = True)
        self.beta = nn.Parameter(torch.zeros(self.num_subsets), requires_grad = True)
        #self.gamma = nn.Parameter(torch.tensor(1.0), requires_grad = True)
        self.inter_act = nn.Tanh()
        self.intra_act = nn.Softmax(-2)
        self.pre = nn.Sequential(
            nn.Conv2d(dim_in, self.mid_channels * self.num_subsets, 1),
            nn.BatchNorm2d(self.mid_channels * self.num_subsets), nn.ReLU())
        
        self.post = nn.Conv2d(self.mid_channels * self.num_subsets, dim, 1)
        self.conv1 = nn.Conv2d(dim_in, self.mid_channels * self.num_subsets, 1)
        self.conv2 = nn.Conv2d(dim_in, self.mid_channels * self.num_subsets, 1)
        self.A = nn.Parameter(A.clone(), requires_grad = True)
        self.act_ov = nn.ReLU()
        if dim_in != dim:
            self.res = nn.Sequential(
                nn.Conv2d(dim_in, dim, 1),
                nn.BatchNorm2d(dim))
        else:
            self.res = lambda x: x 
        #self.dropout = nn.Dropout2d(p=0.1)
        # for debugging
        self._norms = {}


    def forward(self, x, dims) -> Tensor:
        #Mamba
        n, c, t, v = x.size()
        _, _, _, _, m = dims
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v) 
        res = self.res(x)
        # inter + intra
        # N K C T V
        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        # N K C 1 V
        x1 = x1.mean(dim=-2, keepdim=True)
        x2 = x2.mean(dim=-2, keepdim=True)
        # N K C 1 V V = N K C 1 V 1 - N K C 1 1 V
        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        
        # N K C 1 V V
        inter_graph = self.inter_act(diff)
        inter_graph = inter_graph * self.alpha[0]
        # N K C 1 V V = N K C 1 V V + 1 K 1 1 V V
        # N K C 1 V * N K C 1 V = N K 1 1 V V
        intra_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        # N K 1 1 V V
        intra_graph = self.intra_act(intra_graph)
        intra_graph = intra_graph * self.beta[0]
        # N K C 1 V V = N K 1 1 V V + N K C 1 V V

        A = self.A
        A = A[None, :, None, None] 
        
        total_A = inter_graph + intra_graph + A

        # logging
        with torch.no_grad():
            self._norms = {
                "inter_raw": inter_graph.detach().norm(p=2),
                "intra_raw": intra_graph.detach().norm(p=2),
                #"pos": pos_emb.detach().norm(p=2),
                "base": A.detach().norm(p=2),
                "total": total_A.detach().norm(p=2),
            }
          # (N, K, C, V, V)
        total_A = total_A.squeeze(3)
        x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, total_A).contiguous()
        # N K C T V -> N K*C T V
        x = self.post(x.reshape(n, -1, t, v))
        x = self.norm_ov(x)
        x = self.act_ov(x + res)
        #x = self.dropout(x)
        
        return x

    def construct_rpe_hops(self, h1):
        h = [None for _ in range(25)]
        h[0] = torch.eye(25, device=h1.device, dtype=h1.dtype)
        h[1] = h1.clone()
        hops = torch.zeros_like(h[0])
        
        for i in range(2, 25):
            h[i] = h[i - 1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1
        
        for i in range(24, 0, -1):
            diff = h[i] - h[i - 1]
            if diff.any():
                h[i] = diff
                hops += i * h[i]
        
        return hops.long()

class GraphEnt(nn.Module):
    def __init__(self, dim_in, dim, A):
        super().__init__()
        self.conv = unit_gcn(dim_in, dim, A) 
        
        self.dim_in = dim_in
        self.dim = dim

    
    def forward(self, x, dims):
        # N*M, T, V, C
        N, C, T, V = x.size()
        '''N * M - number of video sequences with person number
        C - number of channels (3d position of points)
        T - number of frames
        V - number of skeleton points (25)
        order is: N*M, T, V, C
        '''
        # Sum
        x = self.conv(x, dims).contiguous()
        return x

class GraphTCN(nn.Module):
    def __init__(self, dim_in, dim, A, stride=1, use_mamba = True, d_state = 32, layer = 1, reverse = False, mamba_d=48):
        super().__init__()
        self.dim_in = dim_in
        self.dim = dim
        self.conv = GraphEnt(dim_in, dim, A)
        self.use_mamba = use_mamba
        self.reverse = reverse 
        if use_mamba:            
            self.mamba_d = mamba_d
            self.branch_ch = dim // 4
            
            self.mamba_down = nn.Sequential(
                nn.Conv2d(dim, self.mamba_d, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(self.mamba_d),
                #nn.Dropout2d(0.05)
            )

            self.mamba_up = nn.Sequential(
                nn.Conv2d(self.mamba_d, self.branch_ch*2, kernel_size=1), 
                nn.BatchNorm2d(self.branch_ch*2),
                #nn.Dropout2d(0.1)
            )
            self.mamba_norm = nn.LayerNorm(self.mamba_d) 
            self.mamba = Mamba2(
                d_model=self.mamba_d, 
                d_state=d_state,  
                d_conv=4, 
                headdim=self.mamba_d//4,
                expand=2,
                A_init_range=(1, 16),  # Default
                dt_min=0.001,          # Default
                dt_max=0.1,            # Default
                dt_init_floor=1e-4,
                norm_before_gate = False
                )
            self.mamba_maxpool_branch = nn.Sequential(
                nn.Conv2d(dim, self.branch_ch, kernel_size=1),
                nn.BatchNorm2d(self.branch_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(self.branch_ch)
            )
            self.mamba_direct_branch = nn.Sequential(
                nn.Conv2d(dim, self.branch_ch, kernel_size=1, padding=0, stride=(stride, 1)),
                nn.BatchNorm2d(self.branch_ch),
            )
        else:
            '''self.tcn = MultiScale_TemporalConv(dim, dim, kernel_size=3, stride=stride,
                                            dilations=[1,2],
                                            # residual=True has worse performance in the end
                                            residual=False)'''
        self.act = nn.ReLU(inplace = True)
        if dim_in == dim and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(dim_in, dim, kernel_size=1, stride=stride)
    

    
    def forward(self, x, dims):
        res = self.residual(x)

        #branch_outputs = {}
        
        #gcn
        x = self.conv(x, dims)
        # mamba
        if self.use_mamba:
            direct_out = self.mamba_direct_branch(x)
            maxpool_out = self.mamba_maxpool_branch(x)
            
            # Mamba processing (with optional bidirectional)
            if self.reverse:
                # Forward + reverse (your code, optimized)
                x_rev = torch.flip(x, dims=[2])  # Flip T
                x_combined = torch.cat([x, x_rev], dim=0)  # 2N batch
                mamba_ot = self.mamba_down(x_combined)
            else:
                mamba_ot = self.mamba_down(x)  # Unidirectional
            with torch.cuda.device(x.device):
                n, c, t, v = mamba_ot.size()  # n=2N if bidirectional
                mamba_ot = mamba_ot.permute(0, 3, 2, 1).reshape(n * v, t, c)
                mamba_ot = self.mamba(mamba_ot)
                mamba_ot = self.mamba_norm(mamba_ot)
                mamba_ot = mamba_ot.reshape(n, v, t, c).permute(0, 3, 2, 1)
            
            if self.reverse:
                # Split and align
                mamba_forward, mamba_reverse = mamba_ot.chunk(2, dim=0)
                mamba_reverse = torch.flip(mamba_reverse, dims=[2])
                mamba_ot = (mamba_forward + mamba_reverse) / 2  # Sum for bidirectional context
            else:
                mamba_ot = mamba_ot  # Unidirectional
            
            mamba_out = self.mamba_up(mamba_ot)
            #mamba_out = self.mamba(x)
            branch_outs = [mamba_out, direct_out, maxpool_out]
            x = torch.cat(branch_outs, dim=1)

            ''' for ablation'''
            #branch_outputs['mamba'] = mamba_out.clone().detach()
            #branch_outputs['maxpool'] = maxpool_out.clone().detach()
            #branch_outputs['direct'] = direct_out.clone().detach()
        else:
            #x = self.tcn(x)
            pass
        
        x = self.act(x + res)
        #return x, branch_outputs
        return x

class GraphModel(nn.Module):
    def __init__(self, dim_in, dim, class_num = 60):
        super().__init__()
        self.num_node = 25
        self.num_people = 2
        # line_repr and A
        if class_num == 10:
            self.num_node = 20
            self.num_people = 1
            self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                                (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                                (20, 19)]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            self.neighbor = inward + outward
        else:
            self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                        (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            self.neighbor = inward + outward
        self.A = torch.tensor(get_spatial_graph(self.num_node, self_link, inward, outward), dtype=torch.float32, requires_grad=False)
        #self.A = torch.tensor(random_A(num_filter=8), dtype=torch.float32, requires_grad=False)
        self.l1 = GraphTCN(dim_in, dim, self.A, layer=1, mamba_d=24)
        self.l2 = GraphTCN(dim, dim, self.A, layer=2, mamba_d=24)
        self.l3 = GraphTCN(dim, dim, self.A, layer=3, mamba_d=24)
        self.l4 = GraphTCN(dim, dim, self.A, layer=4, mamba_d=24)
        self.l5 = GraphTCN(dim, dim*2, self.A, layer=5, stride=2)
        self.l6 = GraphTCN(dim*2, dim*2, self.A, layer=6)
        self.l7 = GraphTCN(dim*2, dim*2, self.A, layer=7)
        self.l8 = GraphTCN(dim*2, dim*4, self.A, layer=8, stride=2, mamba_d=64) #64
        self.l9 = GraphTCN(dim*4, dim*4, self.A, layer=9, mamba_d=80) #80
        self.l10 = GraphTCN(dim*4, dim*4, self.A, layer=10, mamba_d=80) #80
        self.layers = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10]

        
        self.fc1 = nn.Linear(dim*4, class_num)
        self.mlp = nn.Sequential(
            self.fc1
        )
        self.data_bn = nn.BatchNorm1d(self.num_people*dim_in*self.num_node) #dim_in
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / class_num))
        bn_init(self.data_bn, 1)
    
    def forward(self, x):
        N, C, T, V, M = x.size()
        dims = x.size()
        # n, c, t, v, m
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        #branch_outputs = [] for ablation
        # N*M, C, T, V
        for i in range(len(self.layers)):
            #x, br = self.layers[i](x, dims)  for ablation
            x = self.layers[i](x, dims)
            #branch_outputs.append(br)  for ablation
        '''
        order is: N*M, C, T, V
        '''
        _, C, T, V = x.size()
        x = x.view(N, M, C, -1)
        
        # order is: N, M, C, T*V
        x = x.mean(3).mean(1)
        x = self.mlp(x)

        #return x, branch_outputs  for ablation
        return x