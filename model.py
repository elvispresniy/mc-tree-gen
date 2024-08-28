import torch
import torch.nn as nn

# Single convololution block
def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(),
    )

# Single transpose convolution layer
def conv_transpose_block(
    in_channels,
    out_channels,
    kernel_size=4,
    stride=2,
    padding=1,
    output_padding=0,
    with_act=True,
):
    modules = [
        nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
    ]
    if with_act:
        modules.append(nn.BatchNorm3d(out_channels))
        modules.append(nn.LeakyReLU())
    return nn.Sequential(*modules)

class VAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(in_channels, 32), # (32, 32, 32)
            conv_block(32, 64),          # (16, 16, 16)
            conv_block(64, 128),         # ( 8,  8,  8)
            conv_block(128, 256),        # ( 4,  4,  4)
            conv_block(256, 512),        # ( 2,  2,  2)
            conv_block(512, 1024),       # ( 1,  1,  1)
        )
        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv_layers(x)
        x = x.reshape(bs, -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return (mu, logvar)
    
class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dims):
        super().__init__()

        self.linear = nn.Linear(latent_dims, 1024) # ( 1,  1)
        self.t_conv_layers = nn.Sequential(
            conv_transpose_block(1024, 512),  # ( 2,  2)
            conv_transpose_block(512, 256),   # ( 4,  4)
            conv_transpose_block(256, 128),   # ( 8,  8)
            conv_transpose_block(128, 64),    # (16, 16)
            conv_transpose_block(64, 32),     # (32, 32)
            conv_transpose_block(32, out_channels, with_act=False), # (64, 64)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)
        x = x.reshape((bs, 1024, 1, 1, 1))
        x = self.t_conv_layers(x)
        x = self.sigmoid(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x):
        # Returns mu, log_var
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Obtain parameters of the normal (Gaussian) distribution
        mu, logvar = self.encode(x)

        # Sample from the distribution
        std = torch.exp(0.5 * logvar)
        z = self.sample(mu, std)

        # Decode the latent point to pixel space
        reconstructed = self.decode(z)

        # Return the reconstructed image, and also the mu and logvar
        # so we can compute a distribution loss
        return reconstructed, mu, logvar

    def sample(self, mu, std):
        # Reparametrization trick
        # Sample from N(0, I), translate and scale
        eps = torch.randn_like(std)
        return mu + eps * std