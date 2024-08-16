import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformer_encoder import TransformerEncoder

class LOD_Multi(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x, bases, d_ff=512, n_heads=8, n_layers=6, dropout=0.1):
        super(LOD_Multi, self).__init__()

        self.N_eig = N_eigen
        self.d_model = N_x
        self.N_time = N_time

        #### time projection
        self.time_projection = nn.Sequential(
                            nn.Linear(init_t, N_time),
                            nn.GELU(),
                            nn.Linear(N_time, N_time)
                        )
        
        #### Encoder
        self.encoder1 = TransformerEncoder(d_model=self.d_model, 
                                          d_ff=d_ff, 
                                          n_heads=n_heads, 
                                          n_layers=n_layers, 
                                          dropout=dropout)
        
        # spatial projection
        self.output_layer1 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        self.latent_out = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        
        self.latent_bases = nn.Parameter(bases)

        ## Positional
        self.pos_encoding = PositionalEncoding(seq_len=init_t, d_model=N_x, n=10000, N_time=N_time)

    # forward
    def forward(self, seq, bases_num):

        _num = seq.size(0)
        _seq = seq.size(1)
        _time = seq.size(2)

        #### time projection
        seq_time_projection = self.time_projection(seq).transpose(1, 2) # (N, 201, 1024)
        seq_time_projection = self.pos_encoding(seq_time_projection)

        # Latent coeff output
        latent_coeff = self.latent_out(seq_time_projection)

        # Encoder
        x_emb = self.encoder1(seq_time_projection, mask=None) + seq_time_projection

        # output
        pred_coeff = self.output_layer1(x_emb) # Best

        # make bases
        pred_pde = torch.zeros((_num, self.N_time, _seq)).to('cuda')
        latent_pde = torch.zeros((_num, self.N_time, _seq)).to('cuda')
        for j in range(len(bases_num)):
            pred_pde[j] = torch.einsum("tn, tns -> ts", pred_coeff[j], self.latent_bases[bases_num[j]]) # (N, T, Seq)
            latent_pde[j] = torch.einsum("tn, tns -> ts", latent_coeff[j], self.latent_bases[bases_num[j]]) # (N, T, Seq)
        
        return pred_pde, latent_pde, pred_coeff, latent_coeff
    
class LOD(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x, bases, d_ff=512, n_heads=8, n_layers=6, dropout=0.1):
        super(LOD, self).__init__()

        self.N_eig = N_eigen
        self.d_model = N_x

        #### Embedding functions
        self.time_projection = nn.Sequential(
                            nn.Linear(init_t, N_time),
                            nn.GELU(),
                            nn.Linear(N_time, N_time)
                        )

        # Encoder
        self.encoder1 = TransformerEncoder(d_model=self.d_model, 
                                          d_ff=d_ff, 
                                          n_heads=n_heads, 
                                          n_layers=n_layers, 
                                          dropout=dropout)
        
        # Output layer
        self.output_layer1 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        self.latent_out = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        
        self.latent_bases = nn.Parameter(bases)

        ## Positional (sincos)
        self.pos_encoding = PositionalEncoding(seq_len=init_t, d_model=N_x, n=10000, N_time=N_time)

    # forward
    def forward(self, seq):

        # Temporal projection + PE
        seq_time_projection = self.time_projection(seq).transpose(1, 2) # (N, 41, 256)
        seq_time_projection = self.pos_encoding(seq_time_projection)

        # Latent coeff output
        latent_coeff = self.latent_out(seq_time_projection)

        # Transformer encoder
        x_emb = self.encoder1(seq_time_projection, mask=None) + seq_time_projection

        # output
        pred_coeff = self.output_layer1(x_emb) # Best

        latent_pde = torch.einsum("btn, tns -> bts", latent_coeff, self.latent_bases) # (N, T, Seq)
        pred_pde = torch.einsum("btn, tns -> bts", pred_coeff, self.latent_bases) # (N, T, Seq)
        
        return pred_pde, latent_pde, pred_coeff, latent_coeff
    
class LOD_wo_PE(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x, bases, d_ff=512, n_heads=8, n_layers=6, dropout=0.1):
        super(LOD_wo_PE, self).__init__()

        self.N_eig = N_eigen
        self.d_model = N_x

        #### Embedding functions
        self.time_projection = nn.Sequential(
                            nn.Linear(init_t, N_time),
                            nn.GELU(),
                            nn.Linear(N_time, N_time)
                        )

        # Encoder
        self.encoder1 = TransformerEncoder(d_model=self.d_model, 
                                          d_ff=d_ff, 
                                          n_heads=n_heads, 
                                          n_layers=n_layers, 
                                          dropout=dropout)
        
        # Output layer
        self.output_layer1 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        self.latent_out = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        
        self.latent_bases = nn.Parameter(bases)

    # forward
    def forward(self, seq):

        # Temporal projection + PE
        seq_time_projection = self.time_projection(seq).transpose(1, 2) # (N, 41, 256)

        # Latent coeff output
        latent_coeff = self.latent_out(seq_time_projection)

        # Transformer encoder
        x_emb = self.encoder1(seq_time_projection, mask=None) + seq_time_projection

        # output
        pred_coeff = self.output_layer1(x_emb) # Best

        latent_pde = torch.einsum("btn, tns -> bts", latent_coeff, self.latent_bases) # (N, T, Seq)
        pred_pde = torch.einsum("btn, tns -> bts", pred_coeff, self.latent_bases) # (N, T, Seq)
        
        return pred_pde, latent_pde, pred_coeff, latent_coeff
    

##################################################################
class PositionalEncoding(nn.Module):
    
    def __init__(self, seq_len, d_model, n, N_time, device='cuda'):
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # encoding : (seq_len, d_model)
        self.encoding = torch.zeros(seq_len, d_model, device=device)
        self.encoding.requires_grad = False
        
        # (seq_len, )
        pos = torch.arange(0, seq_len, device=device)
        # (seq_len, 1)         
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:, ::2] = torch.sin(pos / (n ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (n ** (_2i / d_model)))

        self.encoding = nn.Parameter(self.encoding) # (N, 10, 256)
        self.encoding_layer = nn.Sequential(
            nn.Linear(seq_len, N_time),
            nn.GELU(),
            nn.Linear(N_time, N_time)
        )
        
    def forward(self, x):

        positional_embedding = self.encoding_layer(self.encoding.permute(1,0))
        x = x + positional_embedding.permute(1,0)
        
        return x
