from __future__ import annotations

from typing import Dict

import math

import torch
import torch.nn as nn

from features import (
    EDGE_INTERFACE,
    EDGE_STATIC_DIM,
    NODE_FEATURE_DIM,
    NUM_EDGE_TYPES,
)
try:
    from model.egnn import GaussianRBF, RelationAwareEGNNLayer
except ImportError:
    from egnn import GaussianRBF, RelationAwareEGNNLayer


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


def _segment_softmax(
    scores: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores

    maxima = scores.new_full(
        (num_segments, scores.size(-1)),
        float("-inf"),
    )
    expanded_index = index.view(-1, 1).expand_as(scores)
    if hasattr(maxima, "scatter_reduce_"):
        maxima.scatter_reduce_(
            0,
            expanded_index,
            scores,
            reduce="amax",
            include_self=True,
        )
    else:
        for segment in range(num_segments):
            mask = index == segment
            if torch.any(mask):
                maxima[segment] = scores[mask].max(dim=0).values

    shifted = scores - maxima[index]
    exponentials = torch.exp(shifted)
    denominator = _scatter_add(
        exponentials,
        index,
        num_segments,
    )
    return exponentials / denominator[index].clamp_min(1e-12)


def _safe_segment_mean(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((num_graphs, x.size(-1)))
    summed = _scatter_add(x, batch, num_graphs)
    count = _scatter_add(
        torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype),
        batch,
        num_graphs,
    )
    return summed / count.clamp_min(1.0)


def _safe_segment_max(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((num_graphs, x.size(-1)))

    out = x.new_full((num_graphs, x.size(-1)), float("-inf"))
    expanded_index = batch.view(-1, 1).expand(-1, x.size(-1))
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(
            0,
            expanded_index,
            x,
            reduce="amax",
            include_self=True,
        )
    else:
        # 较旧 PyTorch 的保守兼容路径。
        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            if torch.any(mask):
                out[graph_idx] = x[mask].max(dim=0).values

    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


class GraphAttentionPool(nn.Module):
    """ligand"""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_dim, max(emb_dim // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(emb_dim // 2, 16), 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        *,
        return_attention: bool = False,
    ):
        if h.numel() == 0:
            pooled = h.new_zeros((num_graphs, h.size(-1)))
            if return_attention:
                empty = h.new_zeros((0, 1))
                return pooled, empty, empty
            return pooled

        raw_scores = self.score(h)
        weights = _segment_softmax(raw_scores, batch, num_graphs)
        pooled = _scatter_add(
            weights * h,
            batch,
            num_graphs,
        )
        if return_attention:
            return pooled, raw_scores, weights
        return pooled


class LigandConditionedPocketPool(nn.Module):
    """pocket"""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.scale = emb_dim ** -0.5

    def forward(
        self,
        pocket_h: torch.Tensor,
        pocket_batch: torch.Tensor,
        ligand_graph_h: torch.Tensor,
        num_graphs: int,
        *,
        return_attention: bool = False,
    ):
        if pocket_h.numel() == 0:
            pooled = pocket_h.new_zeros((num_graphs, pocket_h.size(-1)))
            if return_attention:
                empty = pocket_h.new_zeros((0, 1))
                return pooled, empty, empty
            return pooled

        q = self.query(ligand_graph_h)[pocket_batch]
        k = self.key(pocket_h)
        v = self.value(pocket_h)
        raw_scores = (q * k).sum(dim=-1, keepdim=True) * self.scale
        weights = _segment_softmax(raw_scores, pocket_batch, num_graphs)
        pooled = _scatter_add(
            weights * v,
            pocket_batch,
            num_graphs,
        )
        if return_attention:
            return pooled, raw_scores, weights
        return pooled


class InterfacePairAttentionPool(nn.Module):
    """interface"""

    def __init__(
        self,
        emb_dim: int,
        rbf_dim: int,
        rbf_stop: float,
        interface_cutoff: float = 6.0,
    ):
        super().__init__()
        self.interface_cutoff = float(interface_cutoff)
        self.rbf = GaussianRBF(
            num_centers=rbf_dim,
            start=0.0,
            stop=rbf_stop,
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(emb_dim * 2 + rbf_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(emb_dim, max(emb_dim // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(emb_dim // 2, 16), 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos_reference: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        lig_mask: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        *,
        return_attention: bool = False,
    ):
        source, target = edge_index

     
        pair_mask = (
            (edge_type == EDGE_INTERFACE)
            & lig_mask[source]
            & (~lig_mask[target])
        )
        ligand_index = source[pair_mask]
        protein_index = target[pair_mask]

        if ligand_index.numel() == 0:
            zero = h.new_zeros((num_graphs, h.size(-1)))
            if return_attention:
                details = {
                    "ligand_node_indices": ligand_index,
                    "protein_node_indices": protein_index,
                    "raw_scores": h.new_zeros((0, 1)),
                    "weights": h.new_zeros((0, 1)),
                    "distances": h.new_zeros((0,)),
                    "distance_gates": h.new_zeros((0,)),
                }
                return zero, zero, zero, details
            return zero, zero, zero

        distances = torch.linalg.vector_norm(
            pos_reference[ligand_index] - pos_reference[protein_index],
            dim=-1,
        ).clamp_min(1e-6)
        rbf = self.rbf(distances)

        pair_h = self.pair_encoder(
            torch.cat(
                [h[ligand_index], h[protein_index], rbf],
                dim=-1,
            )
        )
        pair_batch = batch[ligand_index]

        scaled_distance = (distances / self.interface_cutoff).clamp(0.0, 1.0)
        distance_gates = 0.5 * (
            torch.cos(math.pi * scaled_distance) + 1.0
        )
        distance_gates = distance_gates * (
            distances < self.interface_cutoff
        ).to(distances.dtype)
        raw_scores = self.score(pair_h)
        gated_scores = raw_scores + torch.log(
            distance_gates.clamp_min(1e-8)
        ).unsqueeze(-1)
        weights = _segment_softmax(gated_scores, pair_batch, num_graphs)

        attention_pool = _scatter_add(
            weights * pair_h,
            pair_batch,
            num_graphs,
        )
        mean_pool = _safe_segment_mean(pair_h, pair_batch, num_graphs)
        max_pool = _safe_segment_max(pair_h, pair_batch, num_graphs)

        if return_attention:
            details = {
                "ligand_node_indices": ligand_index,
                "protein_node_indices": protein_index,
                "raw_scores": raw_scores,
                "weights": weights,
                "distances": distances,
                "distance_gates": distance_gates,
            }
            return attention_pool, mean_pool, max_pool, details
        return attention_pool, mean_pool, max_pool


class ParamMatchMLP(nn.Module):
    """Parameter-matched placeholder for a removed readout track.

    Maps a pooled graph feature (size ``hidden_dim``) to a track embedding of the
    same size. Hidden width is chosen so total trainable parameters approximate
    ``target_params`` (the removed track's attention + projector count).
    """

    def __init__(self, hidden_dim: int, target_params: int, dropout_rate: float = 0.0):
        super().__init__()
        # Linear(H,M)+LN(M)+Linear(M,H)+LN(H) ≈ 2*H*M + 3*M + 3*H
        denom = max(2 * hidden_dim + 3, 1)
        mid = max(int(round((target_params - 3 * hidden_dim) / denom)), 1)
        # Fine-tune mid by ±1 search so realized count is closest to target.
        best_mid, best_err = mid, abs(_param_match_mlp_count(hidden_dim, mid) - target_params)
        for cand in (mid - 2, mid - 1, mid + 1, mid + 2):
            if cand < 1:
                continue
            err = abs(_param_match_mlp_count(hidden_dim, cand) - target_params)
            if err < best_err:
                best_mid, best_err = cand, err
        mid = best_mid
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.target_params = int(target_params)
        self.mid = int(mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _param_match_mlp_count(hidden_dim: int, mid: int) -> int:
    return 2 * hidden_dim * mid + 3 * mid + 3 * hidden_dim


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class PureGeometricAffinityNetwork(nn.Module):
    """network"""

    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        edge_static_dim: int = EDGE_STATIC_DIM,
        num_edge_types: int = NUM_EDGE_TYPES,
        hidden_dim: int = 128,
        num_layers: int = 4,
        rbf_dim: int = 16,
        interface_cutoff: float = 6.0,
        protein_spatial_cutoff: float = 6.0,
        dropout_rate: float = 0.15,
        coord_update: bool = False,
        max_coord_step: float = 0.1,
        use_track_a: bool = True,
        use_track_b: bool = True,
        use_track_c: bool = True,
        param_match: bool = False,
        param_match_track: str | None = None,
    ):
        super().__init__()

        self.use_track_a = bool(use_track_a)
        self.use_track_b = bool(use_track_b)
        self.use_track_c = bool(use_track_c)
        self.param_match = bool(param_match)
        self.param_match_track = (
            str(param_match_track).lower() if param_match_track is not None else None
        )
        if self.param_match:
            if self.param_match_track not in {"a", "b", "c"}:
                raise ValueError("param_match_track must be one of {'a','b','c'}")
            # The matched track must be disabled; MLP replaces that readout branch.
            if self.param_match_track == "a" and self.use_track_a:
                raise ValueError("param_match track A requires use_track_a=False")
            if self.param_match_track == "b" and self.use_track_b:
                raise ValueError("param_match track B requires use_track_b=False")
            if self.param_match_track == "c" and self.use_track_c:
                raise ValueError("param_match track C requires use_track_c=False")

        # Fusion width: remaining real tracks + optional param-match placeholder.
        active_tracks = sum(
            [self.use_track_a, self.use_track_b, self.use_track_c]
        ) + int(self.param_match)
        if active_tracks == 0:
            raise ValueError("At least one readout track must be enabled")

        def make_node_encoder():
            return nn.Sequential(
                nn.Linear(node_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

     
        self.ligand_encoder = make_node_encoder()
        self.protein_encoder = make_node_encoder()

   
        message_rbf_cutoff = max(
            float(interface_cutoff),
            float(protein_spatial_cutoff),
        )

        self.egnn_layers = nn.ModuleList(
            [
                RelationAwareEGNNLayer(
                    emb_dim=hidden_dim,
                    edge_static_dim=edge_static_dim,
                    hidden_dim=hidden_dim,
                    num_edge_types=num_edge_types,
                    rbf_dim=rbf_dim,
                    rbf_start=0.0,
                    rbf_stop=message_rbf_cutoff,
                    coord_update=coord_update,
                    max_coord_step=max_coord_step,
                    dropout=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        if self.use_track_a:
            self.ligand_attention = GraphAttentionPool(hidden_dim)
            self.track_a_projector = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

        if self.use_track_b:
            self.pocket_attention = LigandConditionedPocketPool(hidden_dim)
            self.track_b_projector = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

        if self.use_track_c:
            self.interface_attention = InterfacePairAttentionPool(
                emb_dim=hidden_dim,
                rbf_dim=rbf_dim,
                rbf_stop=interface_cutoff,
                interface_cutoff=interface_cutoff,
            )
            self.track_c_projector = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

        # Parameter-matched control: replace a removed track with an MLP of
        # comparable size. Target counts are measured from the corresponding
        # track modules on a temporary full-track skeleton (same hidden_dim).
        if self.param_match:
            target = self._estimate_track_param_count(
                track=self.param_match_track,
                hidden_dim=hidden_dim,
                rbf_dim=rbf_dim,
                interface_cutoff=interface_cutoff,
                dropout_rate=dropout_rate,
            )
            self.param_match_mlp = ParamMatchMLP(
                hidden_dim=hidden_dim,
                target_params=target,
                dropout_rate=dropout_rate,
            )
            self.param_match_target = int(target)
        else:
            self.param_match_mlp = None
            self.param_match_target = 0

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * active_tracks, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 32)),
            nn.SiLU(),
            nn.Linear(max(hidden_dim // 2, 32), 1),
        )

    @staticmethod
    def _estimate_track_param_count(
        track: str,
        hidden_dim: int,
        rbf_dim: int,
        interface_cutoff: float,
        dropout_rate: float,
    ) -> int:
        """Count parameters of the removed track's attention + projector."""
        if track == "a":
            modules = nn.ModuleList(
                [
                    GraphAttentionPool(hidden_dim),
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                    ),
                ]
            )
        elif track == "b":
            modules = nn.ModuleList(
                [
                    LigandConditionedPocketPool(hidden_dim),
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                    ),
                ]
            )
        else:
            modules = nn.ModuleList(
                [
                    InterfacePairAttentionPool(
                        emb_dim=hidden_dim,
                        rbf_dim=rbf_dim,
                        rbf_stop=interface_cutoff,
                        interface_cutoff=interface_cutoff,
                    ),
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                    ),
                ]
            )
        return _count_params(modules)

    def forward(self, data, return_interpretability: bool = False):
        x = data.x
        pos_reference = data.pos
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_type = data.edge_type
        lig_mask = data.lig_mask

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            num_graphs = int(
                getattr(data, "num_graphs", int(batch.max().item()) + 1)
            )
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            num_graphs = 1

        ligand_counts = _scatter_add(
            lig_mask.to(dtype=x.dtype).unsqueeze(-1),
            batch,
            num_graphs,
        ).squeeze(-1)
        protein_counts = _scatter_add(
            (~lig_mask).to(dtype=x.dtype).unsqueeze(-1),
            batch,
            num_graphs,
        ).squeeze(-1)
        if torch.any(ligand_counts == 0):
            raise ValueError("At least one graph has no ligand atoms")
        if torch.any(protein_counts == 0):
            raise ValueError("At least one graph has no protein atoms")

        h = x.new_zeros((x.size(0), self.ligand_encoder[0].out_features))
        h[lig_mask] = self.ligand_encoder(x[lig_mask])
        h[~lig_mask] = self.protein_encoder(x[~lig_mask])

        for layer in self.egnn_layers:
            h, pos = layer(
                h,
                pos,
                edge_index,
                edge_attr,
                edge_type,
            )

        ligand_h = h[lig_mask]
        ligand_batch = batch[lig_mask]
        protein_h = h[~lig_mask]
        protein_batch = batch[~lig_mask]

        ligand_mean = _safe_segment_mean(
            ligand_h,
            ligand_batch,
            num_graphs,
        )
        track_embeddings = []
        interpretability: Dict[str, dict] = {}
        protein_mean = _safe_segment_mean(protein_h, protein_batch, num_graphs)

        # Keep fusion order A -> B -> C (param-match placeholder occupies the
        # removed track's slot so readout width matches the full model).
        if self.use_track_a:
            ligand_max = _safe_segment_max(
                ligand_h,
                ligand_batch,
                num_graphs,
            )
            if return_interpretability:
                attn_pool, raw_scores, weights = self.ligand_attention(
                    ligand_h,
                    ligand_batch,
                    num_graphs,
                    return_attention=True,
                )
                interpretability["ligand"] = {
                    "node_indices": torch.where(lig_mask)[0],
                    "raw_scores": raw_scores.detach(),
                    "weights": weights.detach(),
                }
            else:
                attn_pool = self.ligand_attention(
                    ligand_h,
                    ligand_batch,
                    num_graphs,
                )
            track_a = self.track_a_projector(
                torch.cat([attn_pool, ligand_mean, ligand_max], dim=-1)
            )
            track_embeddings.append(track_a)
        elif self.param_match and self.param_match_track == "a":
            track_embeddings.append(self.param_match_mlp(ligand_mean))

        if self.use_track_b:
            protein_max = _safe_segment_max(
                protein_h,
                protein_batch,
                num_graphs,
            )
            if return_interpretability:
                pocket_pool, raw_scores, weights = self.pocket_attention(
                    protein_h,
                    protein_batch,
                    ligand_mean,
                    num_graphs,
                    return_attention=True,
                )
                interpretability["pocket"] = {
                    "node_indices": torch.where(~lig_mask)[0],
                    "raw_scores": raw_scores.detach(),
                    "weights": weights.detach(),
                }
            else:
                pocket_pool = self.pocket_attention(
                    protein_h,
                    protein_batch,
                    ligand_mean,
                    num_graphs,
                )
            track_b = self.track_b_projector(
                torch.cat(
                    [pocket_pool, protein_mean, protein_max],
                    dim=-1,
                )
            )
            track_embeddings.append(track_b)
        elif self.param_match and self.param_match_track == "b":
            track_embeddings.append(self.param_match_mlp(protein_mean))

        if self.use_track_c:
            if return_interpretability:
                interface_pool, interface_mean, interface_max, details = (
                    self.interface_attention(
                        h,
                        pos_reference,
                        edge_index,
                        edge_type,
                        lig_mask,
                        batch,
                        num_graphs,
                        return_attention=True,
                    )
                )
                interpretability["interface_pairs"] = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in details.items()
                }
            else:
                interface_pool, interface_mean, interface_max = (
                    self.interface_attention(
                        h,
                        pos_reference,
                        edge_index,
                        edge_type,
                        lig_mask,
                        batch,
                        num_graphs,
                    )
                )
            track_c = self.track_c_projector(
                torch.cat(
                    [interface_pool, interface_mean, interface_max],
                    dim=-1,
                )
            )
            track_embeddings.append(track_c)
        elif self.param_match and self.param_match_track == "c":
            # Cheap pooled context (ligand+pocket mean) for the placeholder.
            track_embeddings.append(
                self.param_match_mlp(0.5 * (ligand_mean + protein_mean))
            )

        graph_embedding = self.readout(torch.cat(track_embeddings, dim=-1))
        affinity = self.regressor(graph_embedding).squeeze(-1)

        if return_interpretability:
            return affinity, interpretability
        return affinity
