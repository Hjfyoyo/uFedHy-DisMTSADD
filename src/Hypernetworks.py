from collections import OrderedDict
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.utils import spectral_norm
import random
from algorithms.Transformer.SCNorTransformer import SCNorTransformer

class ViTHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, featrue, hidden_dim, dim, client_sample, heads=8, dim_head=16, n_hidden=1, depth=6,
                 spec_norm=False):
        super(ViTHyper, self).__init__()
        self.feat = featrue
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.to_qkv_value_list = nn.ModuleList([])
        self.to_qkv_bias_list = nn.ModuleList([])
        self.to_key_projection_weight = nn.ModuleList([])
        self.to_key_projection_bias = nn.ModuleList([])
        self.to_value_projection_weight = nn.ModuleList([])
        self.to_value_projection_bias = nn.ModuleList([])
        self.to_out_projection_weight = nn.ModuleList([])
        self.to_out_projection_bias = nn.ModuleList([])
        self.to_attn_layer_conv1_weight_list = nn.ModuleList([])
        self.to_attn_layer_conv1_bias_list = nn.ModuleList([])
        self.to_attn_layer_conv2_weight_list = nn.ModuleList([])
        self.to_attn_layer_conv2_bias_list = nn.ModuleList([])
        self.to_attn_layer_norm1_weight_list = nn.ModuleList([])
        self.to_attn_layer_norm1_bias_list = nn.ModuleList([])
        self.to_attn_layer_norm2_weight_list = nn.ModuleList([])
        self.to_attn_layer_norm2_bias_list = nn.ModuleList([])

        self.embed_weight = nn.Linear(hidden_dim, self.feat * self.inner_dim)
        self.embed_bias = nn.Linear(hidden_dim, self.inner_dim)
        self.encoder_weight = nn.Linear(hidden_dim, self.inner_dim)
        self.encoder_bias = nn.Linear(hidden_dim, self.inner_dim)
        self.projection_weight = nn.Linear(hidden_dim, self.feat * self.inner_dim)
        self.projection_bias = nn.Linear(hidden_dim, self.feat)

        for d in range(self.depth):
            to_qkv_value = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_akv_bias = nn.Linear(hidden_dim, self.inner_dim)

            key_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            key_bias = nn.Linear(hidden_dim, self.inner_dim)
            value_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            value_bias = nn.Linear(hidden_dim, self.inner_dim)
            out_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            out_bias = nn.Linear(hidden_dim, self.inner_dim)

            to_atten_conv1_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_atten_conv1_bias = nn.Linear(hidden_dim,  self.inner_dim)
            to_atten_conv2_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_atten_conv2_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm1_weight = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm1_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm2_weight = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm2_bias = nn.Linear(hidden_dim, self.inner_dim)
            self.to_qkv_value_list.append(to_qkv_value)
            self.to_qkv_bias_list.append(to_akv_bias)
            self.to_key_projection_weight.append(key_weight)
            self.to_key_projection_bias.append(key_bias)
            self.to_value_projection_weight.append(value_weight)
            self.to_value_projection_bias.append(value_bias)
            self.to_out_projection_weight.append(out_weight)
            self.to_out_projection_bias.append(out_bias)
            self.to_attn_layer_conv1_weight_list.append(to_atten_conv1_weight)
            self.to_attn_layer_conv1_bias_list.append(to_atten_conv1_bias)
            self.to_attn_layer_conv2_weight_list.append(to_atten_conv2_weight)
            self.to_attn_layer_conv2_bias_list.append(to_atten_conv2_bias)
            self.to_attn_layer_norm1_weight_list.append(to_atten_norm1_weight)
            self.to_attn_layer_norm1_bias_list.append(to_atten_norm1_bias)
            self.to_attn_layer_norm2_weight_list.append(to_atten_norm2_weight)
            self.to_attn_layer_norm2_bias_list.append(to_atten_norm2_bias)

    def forward(self, idx, test):
        weights = 0
        emd = self.embeddings(idx)
        # print(emd.shape)
        features = self.mlp(emd)
        # print(features.shape)
        if test == False:
            weights = [OrderedDict() for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(-1,self.inner_dim,self.dim)
                layer_d_qkv_bias_hyper = self.to_qkv_bias_list[d]
                layer_d_qkv_bias = layer_d_qkv_bias_hyper(features).view(-1,self.dim)

                layer_key_weight_hyper = self.to_key_projection_weight[d]
                layer_key_weight = layer_key_weight_hyper(features).view(-1, self.inner_dim, self.dim)
                layer_key_bias_hyper = self.to_key_projection_bias[d]
                layer_key_bias = layer_key_bias_hyper(features).view(-1, self.dim)
                layer_value_weight_hyper = self.to_value_projection_weight[d]
                layer_value_weight = layer_value_weight_hyper(features).view(-1, self.inner_dim, self.dim)
                layer_value_bias_hyper = self.to_value_projection_bias[d]
                layer_value_bias = layer_value_bias_hyper(features).view(-1, self.dim)
                layer_out_weight_hyper = self.to_out_projection_weight[d]
                layer_out_weight = layer_out_weight_hyper(features).view(-1, self.inner_dim, self.dim)
                layer_out_bias_hyper = self.to_out_projection_bias[d]
                layer_out_bias = layer_out_bias_hyper(features).view(-1, self.dim)

                layer_atten_conv1_weight_hyper = self.to_attn_layer_conv1_weight_list[d]
                layer_atten_conv1_weight = layer_atten_conv1_weight_hyper(features).view(-1, self.inner_dim, self.dim, 1)
                layer_atten_conv1_bias_hyper = self.to_attn_layer_conv1_bias_list[d]
                layer_atten_conv1_bias = layer_atten_conv1_bias_hyper(features).view(-1, self.dim)
                layer_atten_conv2_weight_hyper = self.to_attn_layer_conv2_weight_list[d]
                layer_atten_conv2_weight = layer_atten_conv2_weight_hyper(features).view(-1, self.inner_dim, self.dim, 1)
                layer_atten_conv2_bias_hyper = self.to_attn_layer_conv2_bias_list[d]
                layer_atten_conv2_bias = layer_atten_conv2_bias_hyper(features).view(-1, self.dim)
                layer_atten_norm1_weight_hyper = self.to_attn_layer_norm1_weight_list[d]
                layer_atten_norm1_weight = layer_atten_norm1_weight_hyper(features).view(-1, self.dim)
                layer_atten_norm1_bias_hyper = self.to_attn_layer_norm1_bias_list[d]
                layer_atten_norm1_bias = layer_atten_norm1_bias_hyper(features).view(-1, self.dim)
                layer_atten_norm2_weight_hyper = self.to_attn_layer_norm2_weight_list[d]
                layer_atten_norm2_weight = layer_atten_norm2_weight_hyper(features).view(-1, self.dim)
                layer_atten_norm2_bias_hyper = self.to_attn_layer_norm2_bias_list[d]
                layer_atten_norm2_bias = layer_atten_norm2_bias_hyper(features).view(-1, self.dim)
                for nn in range(self.client_sample):
                    weights[nn]['enc_embedding.value_embedding.weight'] = (self.embed_weight(features).view(-1, self.inner_dim, self.feat))[nn]
                    weights[nn]['enc_embedding.value_embedding.bias'] = (self.embed_bias(features).view(-1, self.inner_dim))[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.query_projection.weight"]= layer_d_qkv_value[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.query_projection.bias"] = layer_d_qkv_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.key_projection.weight"] = layer_key_weight[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.key_projection.bias"] = layer_key_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.value_projection.weight"] = layer_value_weight[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.value_projection.bias"] = layer_value_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.out_projection.weight"] = layer_out_weight[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.out_projection.bias"] = layer_out_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".conv1.weight"] = layer_atten_conv1_weight[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".conv1.bias"] = layer_atten_conv1_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".conv2.weight"] = layer_atten_conv2_weight[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".conv2.bias"] = layer_atten_conv2_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".norm1.weight"] = layer_atten_norm1_weight[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".norm1.bias"] = layer_atten_norm1_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".norm2.weight"] = layer_atten_norm2_weight[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".norm2.bias"] = layer_atten_norm2_bias[nn]
                    if d == (self.depth-1):
                        weights[nn]["encoder.norm.weight"] = (self.encoder_weight(features).view(-1, self.inner_dim))[nn]
                        weights[nn]["encoder.norm.bias"] = (self.encoder_bias(features).view(-1, self.inner_dim))[nn]
                        weights[nn]['projection.weight'] = (self.projection_weight(features).view(-1, self.feat, self.inner_dim))[nn]
                        weights[nn]['projection.bias'] = (self.projection_bias(features).view(-1, self.feat))[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_qkv_bias_hyper = self.to_qkv_bias_list[d]
                layer_d_qkv_bias = layer_d_qkv_bias_hyper(features).view(-1,self.dim)
                layer_atten_conv1_weight_hyper = self.to_attn_layer_conv1_weight_list[d]
                layer_atten_conv1_weight = layer_atten_conv1_weight_hyper(features).view(self.inner_dim, self.dim)
                layer_atten_conv1_bias_hyper = self.to_attn_layer_conv1_bias_list[d]
                layer_atten_conv1_bias = layer_atten_conv1_bias_hyper(features).view(-1, self.dim)
                layer_atten_conv2_weight_hyper = self.to_attn_layer_conv2_weight_list[d]
                layer_atten_conv2_weight = layer_atten_conv2_weight_hyper(features).view(-1, self.inner_dim, self.dim,1)
                layer_atten_conv2_bias_hyper = self.to_attn_layer_conv2_bias_list[d]
                layer_atten_conv2_bias = layer_atten_conv2_bias_hyper(features).view(-1, self.dim)
                weights["encoder.attn_layers."+str(d)+".attention.query_projection.weight"] = layer_d_qkv_value
                weights["encoder.attn_layers." + str(d) + ".attention.query_projection.bias"] = layer_d_qkv_bias
                weights["encoder.attn_layers." + str(d) + ".conv1.weight"] = layer_atten_conv1_weight
                weights["encoder.attn_layers." + str(d) + ".conv1.bias"] = layer_atten_conv1_bias
                weights["encoder.attn_layers." + str(d) + ".conv2.weight"] = layer_atten_conv2_weight
                weights["encoder.attn_layers." + str(d) + ".conv2.bias"] = layer_atten_conv2_bias
        return weights

if __name__ == '__main__':
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    hnet = ViTHyper(n_nodes=10, featrue=33, embedding_dim=100, hidden_dim=100, dim=128, client_sample=2, depth=3)

    trans = iTransformer(33)
    node_id = random.choice(range(10))
    arr = np.arange(10)
    np.random.shuffle(arr)
    selected = arr[:int(10 * 0.25)]
    print(selected)

    # weights = hnet(torch.tensor([selected], dtype=torch.long), False)
    # for key, value in weights[0].items():
    #     print(key, value.shape)
    for param_tensor in trans.state_dict():
        print(param_tensor, "\t", trans.state_dict()[param_tensor].size())
    # current_state = trans.state_dict()


    # trans.load_state_dict(weights[0])
    # print(weights[0])
    # for i in weights[1]:
    #     print(i)
    # total_params = sum(p.numel() for p in hnet.parameters())
    # print(f"模型的参数量为: {total_params}")
    # total_mb = total_params * 4 / (1024 ** 2)  # 一个参数占4个字节，换算成MB
    # print(f"模型的参数量为: {total_mb} MB")