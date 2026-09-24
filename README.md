# SEA-RAFT SmartVector for Nuke

This plugin bakes the connected Nuke image stream to a temporary RGB EXR sequence, runs SEA-RAFT in a separate Python process, writes 32-bit float SmartVector EXRs, and reads the cached vector channels back into the source image stream. Inference never runs during normal Nuke frame evaluation.

The worker uses the official [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) `spring-M` configuration and `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` checkpoint. CUDA is currently required by the Nuke UI.

## Install on Windows

No WSL, virtual machine, system Python 3.10, or separate CUDA Toolkit installation is needed. You need a 64-bit Windows machine with an NVIDIA GPU and compatible driver, Git for Windows, PowerShell, internet access, and Nuke.

Open PowerShell in this repository and run:

```powershell
.\scripts\install_windows.ps1
```

For a dry run, use `-Plan`. The installer downloads the official `uv` binary into `.runtime`, creates an isolated Python 3.10.13 environment there, checks out a pinned SEA-RAFT revision, installs PyTorch 2.2.0 CUDA 12.1 and OpenEXR 3.3.3, downloads the model weights, runs a CUDA and EXR test, and adds this plugin to `%USERPROFILE%\.nuke\init.py`. It records the runtime paths in `smartvector/install_config.json`; newly created nodes use those paths automatically. The `.runtime` directory and config file are gitignored. The first run downloads a multi-gigabyte PyTorch wheel and model weights. Re-running the script reuses the installation.

Restart Nuke and create **SEA-RAFT → SEA-RAFT SmartVector** from the Nodes menu. Connect the image graph, choose a cache directory, and click **Generate SmartVectors**.

Advanced options: `-RuntimeDirectory D:\NukeAIRuntime` places downloads and the environment on another drive; `-SkipNukeRegistration` leaves `init.py` untouched. If Windows blocks running local scripts, use `powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1` from the repository root.

The Group output contains the original input channels plus `smartvector_f01_v01` through `smartvector_f64_v01`, each with `p_u`, `p_v`, `n_u`, `n_v`. Connect it to a stock `VectorDistort`. The `n` channels describe forward flow and `p` backward flow. Absent clip-edge or non-anchor vectors are zero.

## Bake and cache behavior

- **Input** frame range uses an immediately connected Read node's range. For any other graph it uses the project range. **Custom** provides explicit bounds.
- Input EXRs are rendered under `<cache>/.source`. They are a snapshot of the actual upstream result. The worker hashes the baked bytes, so a changed upstream image invalidates an old cache even if filenames and settings stay the same.
- **Inference Max Dimension** sets the largest model input dimension; `0` uses the full input format. Vector magnitudes are scaled back independently on X and Y.
- **Clamp** maps input RGB into `[0, 1]`; **Normalize** uses its 1st and 99th percentiles per frame; **Raw** passes source values through. None changes pixel coordinates. Input is rendered with Nuke Write's `raw` colorspace when available, otherwise `linear`.
- `smartvectors.json` stores the cache fingerprint and settings. Completed EXRs are written atomically and are skipped on retry when the fingerprint matches. A cancelled job keeps completed frames for resume. If settings or input change, enable **Overwrite Existing** or use a different cache folder.
- The cache Read and merge remain in the Nuke script. On reopening, cached EXRs load without running inference. **Refresh Cache** checks the sidecar and the frame sequence.

## Standalone worker

The Nuke plugin writes `job.json` alongside the cache. It can also be run directly with the SEA-RAFT environment:

```text
python -m smartvector.worker /path/to/job.json
```

Run this from the repository root or place it on `PYTHONPATH`. The worker emits `PROGRESS`, `FRAME`, and `DONE` lines for UI integration. `job.json` requires `input`, `output` (each containing `####`), `first`, `last`, `width`, `height`, `max_dimension`, `levels`, and `sea_raft_root`.

## Validation

The installer was run on native Windows with an RTX 4060 Ti and driver 595.79. Its checks passed: PyTorch 2.2.0 CUDA inference through SEA-RAFT, an OpenEXR read/write round trip, and automatic paths in a headless Nuke 17 Group. The full Generate-to-VectorDistort workflow should still be checked on a short shot in the target Nuke project.
