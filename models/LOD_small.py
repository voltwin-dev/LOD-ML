import torch.nn as nn


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



class LOD_small_CFD(nn.Module):
    def __init__(self, init_t, N_eigen, N_time, N_x):
        super(LOD_small_CFD, self).__init__()

        self.N_eig = N_eigen
        self.d_model = N_x
        self.N_time = N_time

        self.time_projection1 = nn.Sequential(
                                        nn.Linear(init_t, N_time),
                                        nn.GELU(),
                                        nn.Linear(N_time, N_time)
                                    )
        
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
        
    # forward
    def forward(self, seq):

        _num = seq.size(0)
        
        # Time projection
        seq_time_projection = self.time_projection1(seq).transpose(1,2) # (N, 63, 128)\

        density_pred_coeff = self.output_layer1(seq_time_projection[:, ::3, :]) # (N, 63, 64=N_eigen)
        pressure_pred_coeff = self.output_layer2(seq_time_projection[:, 1::3, :]) # (N, 63, 64=N_eigen)
        velocity_pred_coeff = self.output_layer3(seq_time_projection[:, 2::3, :]) # (N, 63, 64=N_eigen)

        return (density_pred_coeff, pressure_pred_coeff, velocity_pred_coeff)