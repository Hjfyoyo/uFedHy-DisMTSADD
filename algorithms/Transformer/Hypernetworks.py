from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class ViTHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, featrue, hidden_dim, dim, client_sample, heads=8, dim_head=16, n_hidden=1, depth=62,
                 spec_norm=False):
        super(ViTHyper, self).__init__()
        self.dim = dim
        self.feat = featrue
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

        self.to_attn_layer_conv1_weight_list = nn.ModuleList([])
        self.to_attn_layer_conv1_bias_list = nn.ModuleList([])
        self.to_attn_layer_conv2_weight_list = nn.ModuleList([])
        self.to_attn_layer_conv2_bias_list = nn.ModuleList([])
        self.to_attn_layer_norm1_weight_list = nn.ModuleList([])
        self.to_attn_layer_norm1_bias_list = nn.ModuleList([])
        self.to_attn_layer_norm2_weight_list = nn.ModuleList([])
        self.to_attn_layer_norm2_bias_list = nn.ModuleList([])

        self.to_qkv_value_list = nn.ModuleList([])
        self.to_k_value_list = nn.ModuleList([])
        self.to_v_value_list = nn.ModuleList([])
        self.to_out_projection_weight = nn.ModuleList([])
        self.to_out_projection_bias = nn.ModuleList([])
        self.to_q_bias_list = nn.ModuleList([])
        self.to_k_bias_list = nn.ModuleList([])
        self.to_v_bias_list = nn.ModuleList([])

        self.token_weight = nn.Linear(hidden_dim, self.inner_dim * self.feat * 3)
        self.position_weight = nn.Linear(hidden_dim, self.inner_dim * 5000)
        self.hour_weight = nn.Linear(hidden_dim, self.inner_dim * 24)
        self.weekday_weight = nn.Linear(hidden_dim, self.inner_dim * 7)
        self.day_weight = nn.Linear(hidden_dim, self.inner_dim * 32)
        self.month_weight = nn.Linear(hidden_dim, self.inner_dim * 13)

        self.encoder_weight = nn.Linear(hidden_dim, self.inner_dim)
        self.encoder_bias = nn.Linear(hidden_dim, self.inner_dim)
        self.projection_weight = nn.Linear(hidden_dim, self.feat * self.inner_dim)
        self.projection_bias = nn.Linear(hidden_dim, self.feat)

        for d in range(self.depth):
            to_qkv_value = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_k_value = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_v_value = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_q_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_k_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_v_bias = nn.Linear(hidden_dim, self.inner_dim)
            out_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            out_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_conv1_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_atten_conv1_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_conv2_weight = nn.Linear(hidden_dim, self.dim * self.inner_dim)
            to_atten_conv2_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm1_weight = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm1_bias = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm2_weight = nn.Linear(hidden_dim, self.inner_dim)
            to_atten_norm2_bias = nn.Linear(hidden_dim, self.inner_dim)

            self.to_qkv_value_list.append(to_qkv_value)
            self.to_k_value_list.append(to_k_value)
            self.to_v_value_list.append(to_v_value)
            self.to_q_bias_list.append(to_q_bias)
            self.to_k_bias_list.append(to_k_bias)
            self.to_v_bias_list.append(to_v_bias)
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
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(-1,self.inner_dim,self.dim)
                layer_d_k_value_hyper = self.to_k_value_list[d]
                layer_d_k_value = layer_d_k_value_hyper(features).view(-1,self.inner_dim, self.dim)
                layer_d_v_value_hyper = self.to_v_value_list[d]
                layer_d_v_value = layer_d_v_value_hyper(features).view(-1,self.inner_dim, self.dim)

                layer_q_bias_hyper = self.to_q_bias_list[d]
                layer_q_bias = layer_q_bias_hyper(features).view(-1, self.dim)
                layer_k_bias_hyper = self.to_k_bias_list[d]
                layer_k_bias = layer_k_bias_hyper(features).view(-1, self.dim)
                layer_v_bias_hyper = self.to_v_bias_list[d]
                layer_v_bias = layer_v_bias_hyper(features).view(-1, self.dim)
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
                    weights[nn]['enc_embedding.value_embedding.tokenConv.weight'] = (self.token_weight(features).view(-1, self.inner_dim, self.feat, 3))[nn]
                    weights[nn]['enc_embedding.position_embedding.pe'] = (self.position_weight(features).view(-1, 1, 5000, self.inner_dim))[nn]
                    weights[nn]['enc_embedding.temporal_embedding.hour_embed.emb.weight'] = (self.hour_weight(features).view(-1, 24, self.inner_dim))[nn]
                    weights[nn]['enc_embedding.temporal_embedding.weekday_embed.emb.weight'] = (self.weekday_weight(features).view(-1, 7, self.inner_dim))[nn]
                    weights[nn]['enc_embedding.temporal_embedding.day_embed.emb.weight'] = (self.day_weight(features).view(-1, 32, self.inner_dim))[nn]
                    weights[nn]['enc_embedding.temporal_embedding.month_embed.emb.weight'] = (self.month_weight(features).view(-1, 13, self.inner_dim))[nn]

                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.query_projection.weight"]=layer_d_qkv_value[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.query_projection.bias"] = layer_q_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.key_projection.weight"] = layer_d_k_value[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.key_projection.bias"] = layer_k_bias[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.value_projection.weight"] = layer_d_v_value[nn]
                    weights[nn]["encoder.attn_layers." + str(d) + ".attention.value_projection.bias"] = layer_v_bias[nn]
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

        return weights


if __name__ == '__main__':
    import random
    from Transformer import Transformer
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    hnet = ViTHyper(n_nodes=10, featrue=25, embedding_dim=100, hidden_dim=100, dim=128, client_sample=2, depth=3)

    trans = Transformer(enc_in=25, c_out=25)
    node_id = random.choice(range(10))
    arr = np.arange(10)
    np.random.shuffle(arr)
    selected = arr[:int(10 * 0.25)]
    print(selected)


    weights = hnet(torch.tensor([selected], dtype=torch.long), False)
    # print(weights[0])
    # print(weights[0]['enc_embedding.value_embedding.tokenConv.weight'].shape)
    # for name, param in trans.named_parameters():
    #     print(f"{name}: {param.shape}")
    trans.load_state_dict(weights[0])
    for key, value in weights[0].items():
        print(key, value.shape)
    print(weights[0]['enc_embedding.value_embedding.tokenConv.weight'].shape)
    # for name, param in trans.named_parameters():
    #     print(f"{name}: {param.shape}")
    # weights['transformer.layers.0.0.fn.to_qkv.weight'].shape
    # tran.load_state_dict(weights[0], strict=False)
    # print(weights[0]['transformer.layers.1.0.fn.to_qkv.weight'].shape)
    # print(weights[0]['transformer.layers.2.0.fn.to_qkv.weight'].shape)
    # print(weights['transformer.layers.1.0.fn.to_qkv.weight'].shape)
    # net.load_state_dict()
    # for x in weights:
    #     print(x)
