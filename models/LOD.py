import torch
import torch.nn as nn

from transformer_encoder import TransformerEncoder
from torchtune.modules import RotaryPositionalEmbeddings

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

class LOD_Sorp(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x, bases, d_ff=512, n_heads=8, n_layers=6, dropout=0.1):
        super(LOD_Sorp, self).__init__()

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

        # RoPE
        self.pos_encoding = RotaryPositionalEmbeddings(dim = self.d_model)

    # forward
    def forward(self, seq):

        _batch = seq.size(0)

        # Temporal projection + PE
        seq_time_projection = self.time_projection(seq).transpose(1, 2) # (N, 41, 256)
        seq_time_projection = self.pos_encoding(seq_time_projection.unsqueeze(-2))
        seq_time_projection = seq_time_projection.view(_batch, -1, self.d_model)

        # Latent coeff output
        latent_coeff = self.latent_out(seq_time_projection)

        # Transformer encoder
        x_emb = self.encoder1(seq_time_projection, mask=None) + seq_time_projection

        # output
        pred_coeff = self.output_layer1(x_emb) # Best

        latent_pde = torch.einsum("btn, tns -> bts", latent_coeff, self.latent_bases) # (N, T, Seq)
        pred_pde = torch.einsum("btn, tns -> bts", pred_coeff, self.latent_bases) # (N, T, Seq)
        
        return pred_pde, latent_pde, pred_coeff, latent_coeff

class LOD_CFD(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x, bases, d_ff=512, n_heads=8, n_layers=6, dropout=0.1):
        super(LOD_CFD, self).__init__()

        self.N_eig = N_eigen
        self.d_model = N_x
        self.N_time = N_time

        self.time_projection1 = nn.Sequential(
                                        nn.Linear(init_t, N_time),
                                        nn.GELU(),
                                        nn.Linear(N_time, N_time)
                                    )
        
        #### TODO: Using LLM?
        self.encoder1 = TransformerEncoder(d_model=self.d_model, 
                                          d_ff=d_ff, 
                                          n_heads=n_heads, 
                                          n_layers=n_layers, 
                                          dropout=dropout)
        self.encoder2 = TransformerEncoder(d_model=self.d_model, 
                                          d_ff=d_ff, 
                                          n_heads=n_heads, 
                                          n_layers=n_layers, 
                                          dropout=dropout)
        self.encoder3 = TransformerEncoder(d_model=self.d_model, 
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
        self.output_layer2 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        self.output_layer3 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        
        self.latent_out1 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        self.latent_out2 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        self.latent_out3 = nn.Sequential(
                                nn.Linear(self.d_model, self.d_model//2),
                                nn.GELU(),
                                nn.Linear(self.d_model//2, self.N_eig),
                            )
        
        self.pod_bases = nn.Parameter(bases)
        #self.out1 = nn.Linear(N_x, N_x)
        #self.out2 = nn.Linear(N_x, N_x)

        ## Positional
        self.pos_encoding1 = RotaryPositionalEmbeddings(dim = self.d_model)
        self.pos_encoding2 = RotaryPositionalEmbeddings(dim = self.d_model)
        self.pos_encoding3 = RotaryPositionalEmbeddings(dim = self.d_model)

    # forward
    def forward(self, seq):
        '''
        seq: (N, 10, )
        expected output: (N, 201, N_eig=64)
        '''
        _num = seq.size(0)
        
        # Time projection
        seq_time_projection = self.time_projection1(seq).transpose(1,2) # (N, 63, 128)

        # positional encoding by CFD
        # density, pressure, velocity
        seq_time_projection[:, ::3, :] = self.pos_encoding1(seq_time_projection[:, ::3, :].unsqueeze(-2)).view(_num, -1, self.d_model)
        seq_time_projection[:, 1::3, :] = self.pos_encoding2(seq_time_projection[:, 1::3, :].unsqueeze(-2)).view(_num, -1, self.d_model)
        seq_time_projection[:, 2::3, :] = self.pos_encoding3(seq_time_projection[:, 2::3, :].unsqueeze(-2)).view(_num, -1, self.d_model)

        # Latent coeff output
        density_latent_coeff = self.latent_out1(seq_time_projection[:, ::3, :])
        pressure_latent_coeff = self.latent_out2(seq_time_projection[:, 1::3, :])
        velocity_latent_coeff = self.latent_out3(seq_time_projection[:, 2::3, :])

        # POD-Transformer
        density_emb = self.encoder1(seq_time_projection, mask=None)[:, ::3, :] + seq_time_projection[:, ::3, :]
        pressure_emb = self.encoder2(seq_time_projection, mask=None)[:, 1::3, :] + seq_time_projection[:, 1::3, :]
        velocity_emb = self.encoder3(seq_time_projection, mask=None)[:, 2::3, :] + seq_time_projection[:, 2::3, :]
        density_pred_coeff = self.output_layer1(density_emb)
        pressure_pred_coeff = self.output_layer2(pressure_emb)
        velocity_pred_coeff = self.output_layer3(velocity_emb)

        # POD 연산 (1D-CFD)
        latent_pde = torch.zeros((_num, self.N_time//3, self.d_model, 3)).to('cuda')
        pred_pde = torch.zeros((_num, self.N_time//3, self.d_model, 3)).to('cuda')

        latent_pde[..., 0] = torch.einsum("btn, tns -> bts", density_latent_coeff, self.pod_bases[0]) # (N, T, Seq)
        pred_pde[..., 0] = torch.einsum("btn, tns -> bts", density_pred_coeff, self.pod_bases[0]) # (N, T, Seq)
        
        latent_pde[..., 1] = torch.einsum("btn, tns -> bts", pressure_latent_coeff, self.pod_bases[1]) # (N, T, Seq)
        pred_pde[..., 1] = torch.einsum("btn, tns -> bts", pressure_pred_coeff, self.pod_bases[1]) # (N, T, Seq)

        latent_pde[..., 2] = torch.einsum("btn, tns -> bts", velocity_latent_coeff, self.pod_bases[2]) # (N, T, Seq)
        pred_pde[..., 2] = torch.einsum("btn, tns -> bts", velocity_pred_coeff, self.pod_bases[2]) # (N, T, Seq)
        
        return pred_pde, latent_pde, (density_latent_coeff, pressure_latent_coeff, velocity_latent_coeff), (density_pred_coeff, pressure_pred_coeff, velocity_pred_coeff)

class LOD_2D(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x, bases, d_ff=512, n_heads=8, n_layers=6, dropout=0.1):
        super(LOD_2D, self).__init__()

        '''
        big_model: 1024, 4096, 16, 6, 0.3
        standard: 512, 2048, 8, 6, 0.1

        d_latent and d_model, maybe same! (~=1024)
        '''
        self.N_eig = N_eigen
        self.d_model = N_x

        #### Embedding functions
        self.time_projection = nn.Sequential(
                            nn.Linear(init_t, N_time),
                            nn.GELU(),
                            nn.Linear(N_time, N_time)
                        )
        
        # TransformerEncoder with RoPE
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

        # RoPE
        self.pos_encoding = RotaryPositionalEmbeddings(dim = self.d_model)

    # forward
    def forward(self, seq):

        _batch = seq.size(0)

        # Temporal projection
        seq_time_projection = self.time_projection(seq).transpose(1, 2)
        seq_time_projection = self.pos_encoding(seq_time_projection.unsqueeze(-2)) # (B, T, 1=h, seq)
        seq_time_projection = seq_time_projection.view(_batch, -1, self.d_model)

        # Latent coeff output
        latent_coeff = self.latent_out(seq_time_projection)

        # Transformer encoder with RoPE
        x_emb = self.encoder1(seq_time_projection, mask=None) + seq_time_projection

        # output
        pred_coeff = self.output_layer1(x_emb) # Best

        latent_pde = torch.einsum("btn, tns -> bts", latent_coeff, self.latent_bases) # (N, T, Seq)
        pred_pde = torch.einsum("btn, tns -> bts", pred_coeff, self.latent_bases) # (N, T, Seq)
        
        return pred_pde, latent_pde, pred_coeff, latent_coeff


####################################
# SinCos + Learnable
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