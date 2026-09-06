# Drop-in folder for purchased asset packs

Place each purchased asset pack here as downloaded, one folder per pack,
without renaming or converting anything:

```
assets/source/
  README.md                 (this file; the only tracked file in this directory)
  vretractor/               <- the Fab listing "Operating Volkman Retractor Surgical Instrument" (HQ3DMOD)
    vretractor.fbx
    textures/
      vretractor_metal_Mat_BaseColor.png
      vretractor_metal_Mat_Normal.png
      vretractor_metal_Mat_Roughness.png
      vretractor_metal_Mat_Metallic.png
      vretractor_metal_Mat_AO.png
```

Everything under this directory except this file is git-ignored
(`.gitignore`: `assets/source/*`), and the unit and release tiers check that
no source file, texture or export would be committed. Keep the purchase
record and the terms that applied at purchase with your own records, not in
the repository.

The tool recipes under `configs/assets/<tool>.yaml` name the pack folder,
the file, its expected sha256 and size, and the conversion settings.
`simretract-build-assets --tool <tool>` takes the file from here (unless
`tool_source` in `configs/envs.local.yaml` or `SIMRETRACT_TOOL_SOURCE`
points elsewhere), verifies the checksum, converts it and checks the result
against the recipe fingerprints. The full procedure, the alternative exports
the recipe recognises and the verification numbers: `docs/ASSETS.md`.
