import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborEvidenceEvaluation(nn.Module):

    def __init__(self):
        super().__init__()

        # Input:
        #   1. Cosine similarity
        #   2. Distance-based similarity
        #
        # Output:
        #   Reliability score [0, 1]

        self.mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        center_features,
        neighbor_features
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

        if neighbor_features.shape[-1] != 26:
            raise ValueError(
                "neighbor_features must contain exactly "
                "26 neighbours."
            )

        # --------------------------------------------------
        # Prepare center features
        # --------------------------------------------------

        center = center_features.unsqueeze(-1)

        # Shape:
        # (B,C,D,H,W,1)

        # --------------------------------------------------
        # 1. Cosine similarity
        # --------------------------------------------------

        cosine_similarity = F.cosine_similarity(
            center,
            neighbor_features,
            dim=1,
            eps=1e-8
        )

        # Shape:
        # (B,D,H,W,26)

        # --------------------------------------------------
        # 2. Euclidean distance
        # --------------------------------------------------

        euclidean_distance = torch.norm(
            center - neighbor_features,
            p=2,
            dim=1
        )

        # Shape:
        # (B,D,H,W,26)

        # --------------------------------------------------
        # 3. Convert distance into similarity
        # --------------------------------------------------
        #
        # Smaller distance -> higher similarity
        #
        # exp(-distance) gives:
        #
        # distance = 0       -> 1.0
        # distance increases  -> approaches 0
        #
        # This makes distance direction consistent
        # with cosine similarity.

        distance_similarity = torch.exp(
            -euclidean_distance
        )

        # --------------------------------------------------
        # 4. Normalize cosine similarity to [0,1]
        # --------------------------------------------------
        #
        # Cosine similarity originally lies approximately
        # in [-1,1].
        #
        # Convert it to [0,1]:
        #
        # -1 -> 0
        #  0 -> 0.5
        # +1 -> 1

        cosine_similarity = (
            cosine_similarity + 1.0
        ) / 2.0

        # --------------------------------------------------
        # 5. Combine evidence
        # --------------------------------------------------

        evidence = torch.stack(
            [
                cosine_similarity,
                distance_similarity
            ],
            dim=-1
        )

        # Shape:
        # (B,D,H,W,26,2)

        # --------------------------------------------------
        # 6. Tiny MLP
        # --------------------------------------------------

        reliability = self.mlp(
            evidence
        )

        # Shape:
        # (B,D,H,W,26,1)

        reliability = reliability.squeeze(-1)

        # Shape:
        # (B,D,H,W,26)

        # --------------------------------------------------
        # 7. Convert to reliability score [0,1]
        # --------------------------------------------------

        reliability = self.sigmoid(
            reliability
        )

        return reliability