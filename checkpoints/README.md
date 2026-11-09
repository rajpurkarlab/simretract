# Checkpoints

Model weights are publicly available in the [Hugging Face collection](https://huggingface.co/dddw123/omni-retract-checkpoints-public). No login is required for these downloads.

| Model | Download name | Hugging Face |
|---|---|---|
| SimRetract (with dropout) | `baseline` | [Weights](https://huggingface.co/dddw123/omni-retract-checkpoints-public/tree/main/pi0_e5_step1far1000_plane_ng_bal_drop) |
| Without dropout | `no_dropout` | [Weights](https://huggingface.co/dddw123/omni-retract-checkpoints-public/tree/main/pi0_e5_step1far1000_plane_ng_bal) |
| Without recovery | `no_recovery` | [Weights](https://huggingface.co/dddw123/omni-retract-checkpoints-public/tree/main/pi0_e5_strict_plane_ng_15k_drop) |
| Half data | `half_data` | [Weights](https://huggingface.co/dddw123/omni-retract-checkpoints-public/tree/main/pi0_e5half_step1far500_plane_ng_bal_drop) |
| SmolVLA | `smolvla` | [Weights](https://huggingface.co/dddw123/omni-retract-checkpoints-public/tree/main/smolvla_full_e5_step1far1000_plane_ng_bal_drop) |
| ACT | `act` | [Weights](https://huggingface.co/dddw123/omni-retract-checkpoints-public/tree/main/act_e5_step1far1000_plane_ng_bal_bl_drop) |
| π₀ initialization | `pi0_base` | [Weights](https://huggingface.co/dddw123/omni-retract-checkpoints-public/tree/main/pi0_base_usmd1) |

From the repository root:

```bash
scripts/download_checkpoints.sh baseline
scripts/download_checkpoints.sh baseline --base-weights
```

Downloads are saved to `checkpoints/<name>/`. The downloader verifies file sizes
and SHA-256 values from `configs/hub.yaml`. Keep `config.json`, `norm.json`, and
`model.safetensors` together for each trained policy.

The `no_plane`, `no_mask`, `no_smooth`, and `no_standoff` training presets remain
available, but matching pretrained downloads are not configured. The collection
also contains historical experiments; these are not substitutes for the
corresponding dropout-trained presets. See [model presets](../docs/ABLATIONS.md).

Gated upstream resources, including the PaliGemma tokenizer
([google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224))
and [SAM 3](https://huggingface.co/facebook/sam3), require accepting their
upstream access terms. See [checkpoint terms](../LICENSE-CHECKPOINTS.md).
