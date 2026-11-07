import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class SpectralConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            modes: int
    ) -> None:
        '''
            1D Fourier layer.
            It does FFT, linear transform, and Inverse FFT.    
        '''
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device = x.device,
            dtype = torch.cfloat
        )
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n = x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(
            self,
            modes: int,
            width: int,
            n_layers: int = 4
    ) -> None:
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
  
        self.fc0 = nn.Linear(1, self.width)
        # Fourier blocks
        self.conv_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1, bias = False) for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for n in range(self.n_layers):
            x1 = self.mlp_layers[n](x)
            x2 = self.conv_layers[n](x)
            x = x1 + x2
            x = F.gelu(x)

        x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
