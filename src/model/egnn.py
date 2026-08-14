from __future__ import annotations

import math

import torch
import torch.nn as nn


def _scatter_add(
    values: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    output_shape = (dim_size,) + tuple(values.shape[1:])
    output = values.new_zeros(output_shape)
    expanded_index = index.view(
        index.size(0), *([1] * (values.dim() - 1))
    ).expand_as(values)
    output.scatter_add_(0, expanded_index, values)
    return output


class GaussianRBF(nn.Module):
    """Gaussian radial basis function with smooth cosine cutoff envelope."""

    def __init__(
        self,
        num_centers: int = 16,
        start: float = 0.0,
        stop: float = 6.0,
    ):
        super().__init__()
        if num_centers < 2:
            raise ValueError("num_centers must be >= 2")
        if stop <= start:
            raise ValueError("stop must be greater than start")

        centers = torch.linspace(start, stop, num_centers)
        spacing = float(stop - start) / float(num_centers - 1)
        coefficient = -0.5 / max(spacing * spacing, 1e‑8)
        self.start = float(start)
        self.stop = float(stop)
        self.register_buffer("centers", centers)
        self.register_buffer(
            "coefficient",
            torch.tensor(coefficient, dtype=torch.float32),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(dtype=distances.dtype)
        coefficient = self.coefficient.to(dtype=distances.dtype)
        rbf = torch.exp(
            coefficient * (distances.unsqueeze(-1) - centers) ** 2
        )

        scaled = ((distances - self.start) / (self.stop - self.start)).clamp(0.0, 1.0)
        cutoff = 0.5 * (torch.cos(math.pi * scaled) + 1.0)
        cutoff = cutoff * (
            (distances >= self.start) & (distances < self.stop)
        ).to(distances.dtype)
        return rbf * cutoff.unsqueeze(-1)


class RelationAwareEGNNLayer(nn.Module):
    """
    Multi‑relation EGNN layer.

    - edge_index follows PyG convention: [source, target];
    - distance RBF is recomputed per layer from current coordinates, avoiding conflicts between dynamic coordinates and static geometry;
    - both node messages and coordinate messages are normalized by target‑node degree;
    - coordinate update step size is bounded; coordinate updates can be fully disabled.
    """

    def __init__(
        self,
        emb_dim: int,
        edge_static_dim: int,
        hidden_dim: int = 128,
        num_edge_types: int = 4,
        rbf_dim: int = 16,
        rbf_start: float = 0.0,
        rbf_stop: float = 6.0,
        coord_update: bool = False,
        max_coord_step: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.edge_static_dim = int(edge_static_dim)
        self.num_edge_types = int(num_edge_types)
        self.coord_update = bool(coord_update)
        self.max_coord_step = float(max_coord_step)

        self.rbf = GaussianRBF(
            num_centers=rbf_dim,
            start=rbf_start,
            stop=rbf_stop,
        )

        message_input_dim = emb_dim * 2 + edge_static_dim + rbf_dim
        self.message_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(message_input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, emb_dim),
                )
                for _ in range(num_edge_types)
            ]
        )

        self.node_update = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.node_norm = nn.LayerNorm(emb_dim)

        if self.coord_update:
            self.coord_network = nn.Sequential(
                nn.Linear(emb_dim, max(hidden_dim // 2, 16)),
                nn.SiLU(),
                nn.Linear(max(hidden_dim // 2, 16), 1),
                nn.Tanh(),
            )
            # sigmoid(-2.2) ≈ 0.10; multiplied by max_coord_step for very conservative initialization.
            self.coord_gain_logit = nn.Parameter(torch.tensor(-2.2))

    @staticmethod
    def _degree_normalize(
        values: torch.Tensor,
        target: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        aggregated = _scatter_add(values, target, num_nodes)
        degree = _scatter_add(
            torch.ones(
                (target.numel(), 1),
                device=values.device,
                dtype=values.dtype,
            ),
            target,
            num_nodes,
        )
        return aggregated / degree.clamp_min(1.0)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError("edge_attr and edge_index edge counts do not match")

        source, target = edge_index
        receiver_h = h[target]
        sender_h = h[source]

        coord_diff = pos[target] - pos[source]
        distances = torch.linalg.vector_norm(coord_diff, dim=-1).clamp_min(1e‑6)
        rbf = self.rbf(distances)

        messages = h.new_zeros((edge_index.size(1), self.emb_dim))
        for relation in range(self.num_edge_types):
            mask = edge_type == relation
            if not torch.any(mask):
                continue
            relation_input = torch.cat(
                [
                    receiver_h[mask],
                    sender_h[mask],
                    edge_attr[mask],
                    rbf[mask],
                ],
                dim=-1,
            )
            messages[mask] = self.message_networks[relation](relation_input)

        aggregated = self._degree_normalize(messages, target, h.size(0))
        delta_h = self.node_update(torch.cat([h, aggregated], dim=-1))
        h_new = self.node_norm(h + delta_h)

        if not self.coord_update:
            return h_new, pos

        unit_direction = coord_diff / distances.unsqueeze(-1)
        scalar = self.coord_network(messages)
        gain = torch.sigmoid(self.coord_gain_logit) * self.max_coord_step
        coord_messages = unit_direction * scalar * gain
        coord_delta = self._degree_normalize(
            coord_messages,
            target,
            pos.size(0),
        )
        pos_new = pos + coord_delta
        return h_new, pos_new