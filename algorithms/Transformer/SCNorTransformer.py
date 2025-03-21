import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.Transformer.Transformer_EncDec import Encoder, EncoderLayer
from algorithms.Transformer.SelfAttention_Family import FullAttention, AttentionLayer
from algorithms.Transformer.Embed import DataEmbedding_Conversion
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SCNorTransformer(nn.Module):

    def __init__(self, window):
        super(SCNorTransformer, self).__init__()
        self.win_size = window
        self.pred_len = 0
        self.d_model = 128
        self.d_ff = 128
        self.output_attention = True
        self.dropout = 0.1
        self.e_layers = 3
        self.k = 1
        # Embedding
        self.enc_embedding = DataEmbedding_Conversion(self.win_size, self.d_model, embed_type='fixed', freq='h', dropout=0.1)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,  attention_dropout=0.1,
                                      output_attention=self.output_attention), self.d_model, n_heads=8),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation="relu"
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.projection = nn.Linear(self.d_model, self.win_size, bias=True)

    def forward(self, x_enc):
        # Normalization
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        #
        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        # data_out = dec_out
        # data_out = data_out.squeeze()
        # print(data_out.shape)
        # plt.plot(data_out[:,0,0].cpu().detach().numpy())
        # plt.ylabel('Value')
        # plt.xlabel('Timestamp')
        # plt.savefig(r'E:\pythonProject\FedTAD-main\plots\Normalization\tcm.png')

        others = {}

        return dec_out, attns, others  # [B, L, D]

if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    sns.heatmap(data, cbar=False)
    plt.show()

