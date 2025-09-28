import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                    * self.scale_noise / self.grid_size
            )

            fit_grid = self.grid[:, self.spline_order: -self.spline_order].T  # [grid_size+1, in_features]

            spline_coeffs = self.curve2coeff(fit_grid, noise)  # [out_features, in_features, grid_size + spline_order]

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * spline_coeffs
            )

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):

            left_denom = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_denom = grid[:, k + 1:] - grid[:, 1:(-k)]

            left_denom = torch.where(torch.abs(left_denom) < 1e-8,
                                     torch.ones_like(left_denom) * 1e-8, left_denom)
            right_denom = torch.where(torch.abs(right_denom) < 1e-8,
                                      torch.ones_like(right_denom) * 1e-8, right_denom)

            left = (x - grid[:, : -(k + 1)]) / left_denom * bases[:, :, :-1]
            right = (grid[:, k + 1:] - x) / right_denom * bases[:, :, 1:]
            bases = left + right

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # [in_features, grid_size+1, grid_size + spline_order]
        B = y.transpose(0, 1)  # [in_features, grid_size+1, out_features]

        solution = torch.linalg.lstsq(A, B).solution  # [in_features, grid_size + spline_order, out_features]
        result = solution.permute(2, 0, 1)  # [out_features, in_features, grid_size + spline_order]

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if hasattr(self, 'spline_scaler') and self.spline_scaler is not None
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class DWConv2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x):
        return self.dw(x)


class TokenizedKANBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, grid_size=5):
        super().__init__()
        self.input_adapt = nn.Conv2d(in_channels, embed_dim, kernel_size=1) \
            if in_channels != embed_dim else nn.Identity()

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        self.kan = KANLinear(embed_dim, embed_dim, grid_size=grid_size)

        self.dw = DWConv2d(embed_dim)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU(inplace=True)

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.input_adapt(x)

        z = self.proj(x)
        D = z.size(1)

        tokens = z.permute(0, 2, 3, 1).contiguous()
        tokens_flat = tokens.view(B * H * W, D)

        kan_out = self.kan(tokens_flat)
        kan_out = kan_out.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        dw_out = self.dw(kan_out)
        dw_out = self.bn(dw_out)
        dw_out = self.act(dw_out)

        out = dw_out + z

        out_tokens = out.permute(0, 2, 3, 1).contiguous()
        out_tokens = self.ln(out_tokens)
        out = out_tokens.permute(0, 3, 1, 2).contiguous()
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UKanPaper(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, tok_blocks=2, embed_dim=256):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        bottleneck_channels = 1024 // factor

        if embed_dim is None:
            embed_dim = bottleneck_channels
        self.embed_dim = embed_dim

        tok_blocks_list = []
        for i in range(tok_blocks):
            if i == 0:
                input_ch = bottleneck_channels
            else:
                input_ch = embed_dim
            tok_blocks_list.append(TokenizedKANBlock(input_ch, embed_dim))

        self.tok_blocks = nn.ModuleList(tok_blocks_list)

        self.conv_unproj = nn.Conv2d(embed_dim, bottleneck_channels, kernel_size=1)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        tok = x5
        for block in self.tok_blocks:
            tok = block(tok)

        tok = self.conv_unproj(tok)

        x = self.up1(tok, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
