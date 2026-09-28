import torch
import torch.nn as nn


class ResidualNeighborhoodRefinement(nn.Module):

    def __init__(self, residual_scale=0.1):
        super().__init__()

        # Small residual contribution so that NREM
        # does not strongly disturb the pretrained
        # encoder features.
        self.residual_scale = residual_scale

    def forward(
        self,
        center_features,
        neighbor_features,
        reliability
    ):

        # --------------------------------------------------
        # Input validation
        # --------------------------------------------------

        if center_features.dim() != 5:
            raise ValueError(
                "center_features must have shape "
                "(B,C,D,H,W)"
            )

        if neighbor_features.dim() != 6:
            raise ValueError(
                "neighbor_features must have shape "
                "(B,C,D,H,W,26)"
            )

        if reliability.dim() != 5:
            raise ValueError(
                "reliability must have shape "
                "(B,D,H,W,26)"
            )

        if neighbor_features.shape[-1] != 26:
            raise ValueError(
                "neighbor_features must contain exactly "
                "26 neighbours."
            )

        if reliability.shape[-1] != 26:
            raise ValueError(
                "reliability must contain exactly "
                "26 weights."
            )

        # --------------------------------------------------
        # Expand reliability weights
        # --------------------------------------------------

        # reliability:
        # (B,D,H,W,26)
        #
        # ->
        # (B,1,D,H,W,26)

        weights = reliability.unsqueeze(1)

        # --------------------------------------------------
        # Weighted neighborhood aggregation
        # --------------------------------------------------

        weighted_neighbors = (
            neighbor_features * weights
        )

        # Sum weighted neighbor features
        weighted_sum = weighted_neighbors.sum(dim=-1)

        # Sum reliability weights
        weight_sum = weights.sum(dim=-1)

        # Prevent division by zero
        weight_sum = weight_sum.clamp_min(1e-6)

        # Weighted average of the 26 neighbours
        aggregated_neighbors = (
            weighted_sum / weight_sum
        )

        # --------------------------------------------------
        # Controlled residual refinement
        # --------------------------------------------------

        refined_features = (
            center_features
            + self.residual_scale * aggregated_neighbors
        )

        return refined_features