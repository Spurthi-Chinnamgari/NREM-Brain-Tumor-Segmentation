"""
3D Attention U-Net Model Architecture for Volumetric Medical Image Segmentation.
Ref: Oktay et al., 'Attention U-Net: Learning Where to Look for the Pancreas', MIDL 2018.
"""

from typing import List
import torch
import torch.nn as nn

from .building_blocks import DoubleConv3D, UpBlock3D
from .nrem.nrem import NeighborhoodReliabilityEnhancedModule

class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net with NREM for volumetric multimodal
    brain tumor segmentation.

    NREM is applied to the deepest encoder feature before it
    is passed to the decoder through the deepest skip connection.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: List[int] = None,
        dropout: float = 0.1,
        use_transpose: bool = True,
        use_nrem: bool = True,
    ):
        """
        Args:
            in_channels:
                Number of input MRI modalities.

            out_channels:
                Number of segmentation classes.

            features:
                Channel dimensions at each level.

            dropout:
                Dropout probability.

            use_transpose:
                Whether to use ConvTranspose3d for upsampling.

            use_nrem:
                Whether to use the Neighborhood Reliability
                Enhanced Module on the deepest encoder feature.
        """
        super().__init__()

        if features is None:
            features = [16, 32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_nrem = use_nrem

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        prev_channels = in_channels

        for feature in features[:-1]:
            self.encoder_blocks.append(
                DoubleConv3D(
                    in_channels=prev_channels,
                    out_channels=feature,
                    dropout=dropout,
                )
            )

            prev_channels = feature

        # --------------------------------------------------
        # NREM
        # --------------------------------------------------

        # NREM is applied to the deepest encoder feature.
        #
        # With default input 64x64x64:
        #
        # Encoder 1 -> 32 channels, 64^3
        # Encoder 2 -> 64 channels, 32^3
        # Encoder 3 -> 128 channels, 16^3
        # Encoder 4 -> 256 channels, 8^3
        #
        # Therefore NREM operates on:
        #
        # (B, 256, 8, 8, 8)

        if self.use_nrem:
            self.nrem = NeighborhoodReliabilityEnhancedModule()

        # --------------------------------------------------
        # Bottleneck
        # --------------------------------------------------

        self.bottleneck = DoubleConv3D(
            in_channels=features[-2],
            out_channels=features[-1],
            dropout=dropout,
        )

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        self.decoder_blocks = nn.ModuleList()

        rev_features = list(reversed(features))

        for i in range(len(rev_features) - 1):

            in_ch = rev_features[i]
            out_ch = rev_features[i + 1]

            self.decoder_blocks.append(
                UpBlock3D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    use_transpose=use_transpose,
                    dropout=dropout,
                )
            )

        # --------------------------------------------------
        # Final segmentation layer
        # --------------------------------------------------

        self.final_conv = nn.Conv3d(
            features[0],
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Input tensor of shape
                (B, in_channels, D, H, W)

        Returns:
            Segmentation logits of shape
            (B, out_channels, D, H, W)
        """

        skip_connections = []

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        for i, block in enumerate(self.encoder_blocks):

            x = block(x)

            # --------------------------------------------------
            # Apply NREM only to deepest encoder feature
            # --------------------------------------------------

            if self.use_nrem and i == len(self.encoder_blocks) - 1:
                x = self.nrem(x)

            # Store skip connection
            skip_connections.append(x)

            # Downsample
            x = self.pool(x)

        # --------------------------------------------------
        # Bottleneck
        # --------------------------------------------------

        x = self.bottleneck(x)

        # --------------------------------------------------
        # Reverse skip connections
        # --------------------------------------------------

        skip_connections = list(
            reversed(skip_connections)
        )

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        for i, up_block in enumerate(self.decoder_blocks):

            skip = skip_connections[i]

            x = up_block(
                g=x,
                x=skip,
            )

        # --------------------------------------------------
        # Final segmentation
        # --------------------------------------------------

        logits = self.final_conv(x)

        return logits