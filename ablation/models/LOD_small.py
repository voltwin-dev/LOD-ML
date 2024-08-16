import torch.nn as nn
import torch

class LOD_small(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x):
        super(LOD_small, self).__init__()

        self.N_eig = N_eigen
        self.hidden = N_x

        self.time_projection = nn.Sequential(
                                            nn.Linear(init_t, N_time), # no //2
                                            nn.GELU(),
                                            nn.Linear(N_time, N_time)
                                        )   

        self.latent_out = nn.Sequential(
                                    nn.Linear(self.hidden, self.hidden//2),
                                    nn.GELU(),
                                    nn.Linear(self.hidden//2, N_eigen),
                                )

    def forward(self, seq):
        '''
        seq: (B, S, init_T=10)
        '''

        seq_time_projection = self.time_projection(seq).transpose(1, 2) # (N, T, S)
        output = self.latent_out(seq_time_projection) # (N, T, N_eigen)

        return output
    
class LOD_small_learnable(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x, bases):
        super(LOD_small_learnable, self).__init__()

        self.N_eig = N_eigen
        self.hidden = N_x

        self.time_projection = nn.Sequential(
                                            nn.Linear(init_t, N_time), # no //2
                                            nn.GELU(),
                                            nn.Linear(N_time, N_time)
                                        )   

        self.latent_out = nn.Sequential(
                                    nn.Linear(self.hidden, self.hidden//2),
                                    nn.GELU(),
                                    nn.Linear(self.hidden//2, N_eigen),
                                )
        
        self.latent_bases = nn.Parameter(bases)

    def forward(self, seq):
        '''
        seq: (B, S, init_T=10)
        '''
        
        # New
        seq_time_projection = self.time_projection(seq).transpose(1, 2) # (N, T, S)

        pred_coeff = self.latent_out(seq_time_projection) # (N, T, N_eigen)
        pred_pde = torch.einsum("btn, tns -> bts", pred_coeff, self.latent_bases) # (N, T, Seq)

        return pred_pde, pred_coeff