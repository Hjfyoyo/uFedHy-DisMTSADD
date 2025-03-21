import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.Transformer.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from algorithms.Transformer.SelfAttention_Family import FullAttention, AttentionLayer
from algorithms.Transformer.Embed import DataEmbedding
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Transformer(nn.Module):
    def __init__(self, enc_in, c_out):
        super(Transformer, self).__init__()
        self.enc_in = enc_in
        self.c_out = c_out
        self.n_heads = 8
        self.d_model = 128
        self.d_ff = 128
        self.dropout = 0.1
        self.e_layers = 3
        self.output_attention = True #是否输出注意力分数
        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, embed_type = 'fixed', freq = 'h',
                                           dropout = 0.1)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout = self.dropout,
                                      output_attention = self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=0.1,
                    activation='relu'
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, x_enc):
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        #
        # _, L, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # series = enc_out
        # series = series.squeeze()
        # ax2.plot(series[:,0,0].cpu().detach().numpy())
        dec_out = self.projection(enc_out)
        #
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        # print(dec_out.shape)
        # data_out = dec_out
        # data_out = data_out.squeeze()
        # plt.plot(data_out[:, 0, 0].cpu().detach().numpy())
        # plt.ylabel('Value')
        # plt.xlabel('Timestamp')
        # plt.savefig(r'E:\pythonProject\FedTAD-main\plots\Normalization\nor.png')
        # att = dec_out[:, 0, 0]
        # att = att.unsqueeze(1)
        # sns.heatmap(att.cpu().detach().numpy().T, cmap='viridis', xticklabels=False, yticklabels=False, cbar=False)
        # plt.savefig(r'E:\pythonProject\FedTAD-main\plots\Normalization\Normalization-transfomer-heat.png')

        others = {}
        return dec_out, attns, others  # [B, L, D]

if __name__ == '__main__':
    net = Transformer(enc_in = 33, c_out = 33)
    input = torch.rand((128, 22, 33))
    output, attns, _ = net(input)
    print(output.shape)
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # print(net)
    # print(output.shape, attns[0].shape,attns[1].shape,attns[2].shape)
    # global_weight_collector = list(net.parameters())
    # fed_prox_reg = 0.0
    # for param_index, param in enumerate(net.parameters()):
    #     fed_prox_reg += ((1 / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
    # print(fed_prox_reg)