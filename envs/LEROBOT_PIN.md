# Why lerobot is pinned to a commit

The policy environment installs

```
lerobot[pi,smolvla] @ git+https://github.com/huggingface/lerobot@a07f22e22ce88cddff1f6eddced9ea008fbfc37c
```

with `--no-deps`, followed by the explicit runtime dependencies in `envs/policy.txt`.

## Provenance

* The commit is on `huggingface/lerobot` `main`, version string `0.5.2`, between the
  `v0.5.1` and `v0.6.0` release tags.
* The checkpoints shipped with this repository were trained and evaluated against
  this exact revision of the `pi0`, `pi05`, `pi0_fast`, `smolvla` and `act` policy
  modules and the policy factory. Their `config.json` / `model.safetensors` /
  `norm.json` contract is what `simretract.serve.policy_server` loads.
* Neither adjacent release matches: `v0.5.1` predates fixes in the pi0 image
  embedding path, and `v0.6.0` changes the policy configuration classes and the
  processor pipeline, which alters how a checkpoint's inputs are prepared.
* The upstream files at this commit are byte-identical to what the models were
  trained with, except for a no-op compatibility guard in the pi0 / pi05 image
  embedding (`hasattr(..., "pooler_output")`) that only matters for
  `transformers < 5`; this environment pins `transformers==5.3.0`, so no patch
  is carried.

## Extras

* `pi` adds `transformers==5.3.0` and `scipy` (pi0 / pi05 / pi0_fast).
* `smolvla` adds `num2words` and `accelerate`.
* ACT is part of the core package (ResNet-18 backbone weights come from torchvision).

Because the package is installed with `--no-deps`, the extras only document intent;
the pins in `envs/policy.txt` are authoritative.

## Fast-WAM

The optional Fast-WAM row needs `lerobot==0.6.1` (`envs/fastwam.txt`), which is
incompatible with the pinned policy stack; it lives in its own environment and is
selected through the `fastwam` interpreter role written by `envs/setup.sh`
(`configs/envs.local.yaml`).
