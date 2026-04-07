from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from app.tactical_gnn.graph_builder import FreezeFrameGraph
from app.tactical_gnn.schemas import LABEL_HEADS, TacticalGNNConfig

try:
    from torch_geometric.nn import GATv2Conv  # type: ignore

    HAS_PYG = True
except ImportError:  # pragma: no cover
    GATv2Conv = None
    HAS_PYG = False


def _pool_mean(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    output = torch.zeros((num_graphs, values.shape[-1]), device=values.device, dtype=values.dtype)
    counts = torch.zeros((num_graphs, 1), device=values.device, dtype=values.dtype)
    output.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones((values.shape[0], 1), device=values.device, dtype=values.dtype))
    return output / counts.clamp_min(1.0)


def _pool_max(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    pooled = torch.full(
        (num_graphs, values.shape[-1]),
        fill_value=torch.finfo(values.dtype).min,
        device=values.device,
        dtype=values.dtype,
    )
    for graph_index in range(num_graphs):
        mask = batch == graph_index
        pooled[graph_index] = values[mask].max(dim=0).values if torch.any(mask) else 0.0
    return pooled


def _masked_mean(values: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor, num_graphs: int) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1)
    masked_values = values * expanded_mask
    output = torch.zeros((num_graphs, values.shape[-1]), device=values.device, dtype=values.dtype)
    counts = torch.zeros((num_graphs, 1), device=values.device, dtype=values.dtype)
    output.index_add_(0, batch, masked_values)
    counts.index_add_(0, batch, expanded_mask)
    return output / counts.clamp_min(1.0)


class EdgeAwareGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.msg_proj = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        residual = x
        if edge_index.numel() == 0:
            return self.norm(F.relu(self.self_proj(x)) + residual)
        src_index, dst_index = edge_index
        messages = self.msg_proj(torch.cat([x[src_index], edge_attr], dim=-1))
        aggregated = torch.zeros_like(x)
        counts = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        aggregated.index_add_(0, dst_index, messages)
        counts.index_add_(0, dst_index, torch.ones((messages.shape[0], 1), device=x.device, dtype=x.dtype))
        updated = self.self_proj(x) + aggregated / counts.clamp_min(1.0)
        return self.norm(residual + self.dropout(F.relu(updated)))


class TacticalGNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        config: TacticalGNNConfig,
        label_maps: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_maps = label_maps or config.label_maps
        self.use_pyg = HAS_PYG
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        if self.use_pyg:
            self.graph_layers = nn.ModuleList(
                [
                    GATv2Conv(  # type: ignore[misc]
                        config.hidden_dim,
                        config.hidden_dim,
                        heads=1,
                        dropout=config.dropout,
                        edge_dim=edge_dim,
                    )
                    for _ in range(config.num_layers)
                ]
            )
            self.layer_norms = nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers))
        else:
            self.graph_layers = nn.ModuleList(
                EdgeAwareGraphBlock(config.hidden_dim, edge_dim, config.dropout)
                for _ in range(config.num_layers)
            )
            self.layer_norms = nn.ModuleList()

        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.heads = nn.ModuleDict(
            {head: nn.Linear(config.hidden_dim, len(self.label_maps[head])) for head in LABEL_HEADS}
        )

    def forward(self, graph: FreezeFrameGraph) -> dict[str, torch.Tensor]:
        x = self.dropout(F.relu(self.input_projection(graph.x)))
        if self.use_pyg:
            for layer, layer_norm in zip(self.graph_layers, self.layer_norms):
                residual = x
                x = layer(x, graph.edge_index, graph.edge_attr)
                x = layer_norm(F.relu(x) + residual)
                x = self.dropout(x)
        else:
            for layer in self.graph_layers:
                x = layer(x, graph.edge_index, graph.edge_attr)

        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        mean_pool = _pool_mean(x, graph.batch, num_graphs)
        max_pool = _pool_max(x, graph.batch, num_graphs)
        attacking_pool = _masked_mean(x, graph.batch, graph.teammate_mask, num_graphs)
        defensive_pool = _masked_mean(x, graph.batch, 1.0 - graph.teammate_mask, num_graphs)
        graph_embedding = self.readout(torch.cat([mean_pool, max_pool, attacking_pool, defensive_pool], dim=-1))
        outputs: dict[str, torch.Tensor] = {"embedding": graph_embedding}
        for head_name, head_module in self.heads.items():
            outputs[head_name] = head_module(graph_embedding)
        return outputs


def create_model(
    config: TacticalGNNConfig,
    input_dim: int,
    edge_dim: int,
    label_maps: dict[str, list[str]] | None = None,
) -> TacticalGNNModel:
    return TacticalGNNModel(
        input_dim=input_dim,
        edge_dim=edge_dim,
        config=config,
        label_maps=label_maps,
    )
