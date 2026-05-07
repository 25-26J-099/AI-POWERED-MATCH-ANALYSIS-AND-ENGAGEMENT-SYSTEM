# Tactical GNN Evaluation

- Checkpoint: `checkpoints/tactical_gnn/model.pt`
- Dataset path: `data/tactical_gnn/gnn_synthetic_augmented.jsonl`
- Usable samples: `1855`
- Evaluated samples: `371`
- Dropped samples: `0`
- Evaluation split: `val`
- Active heads: `formation, team_shape, attacking_structure, defensive_block, defensive_shape`

## formation
- Support: `371`
- Accuracy: `0.3046`
- Macro F1: `0.2894`
- Micro F1: `0.3046`

## team_shape
- Support: `371`
- Accuracy: `0.6604`
- Macro F1: `0.5971`
- Micro F1: `0.6604`

## attacking_structure
- Support: `371`
- Accuracy: `0.7332`
- Macro F1: `0.7228`
- Micro F1: `0.7332`

## defensive_block
- Support: `371`
- Accuracy: `0.8356`
- Macro F1: `0.8268`
- Micro F1: `0.8356`

## defensive_shape
- Support: `371`
- Accuracy: `0.5795`
- Macro F1: `0.5702`
- Micro F1: `0.5795`
