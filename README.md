# SEA-RAFT SmartVector for Nuke

This plugin bakes the connected Nuke image stream to a working RGB EXR sequence, runs SEA-RAFT in a separate Python process, and reads the generated vector channels back into the source image stream. The final sequence is rendered by a standard Nuke Write node. Inference never runs during normal frame evaluation.

The worker uses the official [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) `spring-M` configuration and `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` checkpoint. CUDA is currently required by the Nuke UI.

## Install on Windows

No WSL, virtual machine, system Python 3.10, or separate CUDA Toolkit installation is needed. You need a 64-bit Windows machine with an NVIDIA GPU and compatible driver, Git for Windows, PowerShell, internet access, and Nuke.

Open PowerShell in this repository and run:

```powershell
.\scripts\install_windows.ps1
```

For a dry run, use `-Plan`. The installer downloads the official `uv` binary into `.runtime`, creates an isolated Python 3.10.13 environment there, checks out a pinned SEA-RAFT revision, installs PyTorch 2.2.0 CUDA 12.1 and OpenEXR 3.3.3, downloads the model weights, runs a CUDA and EXR test, and adds this plugin to `%USERPROFILE%\.nuke\init.py`. It records the runtime paths in `smartvector/install_config.json`; newly created nodes use those paths automatically. The `.runtime` directory and config file are gitignored. The first run downloads a multi-gigabyte PyTorch wheel and model weights. Re-running the script reuses the installation.

Restart Nuke and create **SEA-RAFT → SEA-RAFT SmartVector** from the Nodes menu. Connect the image graph, click **Export Write**, set the new Write node's EXR sequence path, then click **Generate + Render** on its SEA-RAFT tab. The Write node outputs all channels as 32-bit float EXR. Existing Groups from the previous version are upgraded when a script opens, or with **SEA-RAFT → Upgrade Existing Nodes**.

Advanced options: `-RuntimeDirectory D:\NukeAIRuntime` places downloads and the environment on another drive; `-SkipNukeRegistration` leaves `init.py` untouched. If Windows blocks running local scripts, use `powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1` from the repository root.

The Group output contains the original input channels plus `smartvector_f01_v01` through `smartvector_f64_v01`, each with `p_u`, `p_v`, `n_u`, `n_v`. Connect it to a stock `VectorDistort`. The `n` channels describe forward flow and `p` backward flow. Absent clip-edge or non-anchor vectors are zero.

## Write and working files

- **Input** frame range uses an immediately connected Read node's range. For any other graph it uses the project range. **Custom** provides explicit bounds.
- **Generate + Render** first uses a temporary Write to materialize the actual upstream image, runs the external worker, then renders the final output through the connected Write. Nuke cannot start another `nuke.execute` from a Write `beforeRender` callback, so use the dedicated button for a fresh bake and render.
- Working EXRs and metadata live in a hidden `.sea_raft_work/<node-id>` folder beside the final Write path. They are intermediate files, and the Write path is the deliverable. The worker hashes the baked input bytes to detect changes on subsequent preparations.
- **Inference Max Dimension** sets the largest model input dimension; `0` uses the full input format. Vector magnitudes are scaled back independently on X and Y.
- **Clamp** maps input RGB into `[0, 1]`; **Normalize** uses its 1st and 99th percentiles per frame; **Raw** passes source values through. None changes pixel coordinates. Input is rendered with Nuke Write's `raw` colorspace when available, otherwise `linear`.
- `smartvectors.json` stores the working fingerprint and settings. Completed working EXRs are written atomically and can be skipped on retry when the fingerprint matches. A cancelled job keeps completed frames for resume. If settings or input change, enable **Overwrite Existing** or choose a new Write path.
- The internal Read and merge remain in the Nuke script so prepared vectors can be viewed after reopening without another inference pass. **Refresh Status** checks the sidecar and frame sequence.

## Standalone worker

The Nuke plugin writes `job.json` alongside the cache. It can also be run directly with the SEA-RAFT environment:

```text
python -m smartvector.worker /path/to/job.json
```

Run this from the repository root or place it on `PYTHONPATH`. The worker emits `PROGRESS`, `FRAME`, and `DONE` lines for UI integration. `job.json` requires `input`, `output` (each containing `####`), `first`, `last`, `width`, `height`, `max_dimension`, `levels`, and `sea_raft_root`.

## Validation

The installer was run on native Windows with an RTX 4060 Ti and driver 595.79. A headless Nuke 17 test passed the full two-frame **Generate + Render** path through a Write node; the final EXR contained RGB and all 28 SmartVector channels as 32-bit float. A short shot should still be checked visually in stock `VectorDistort`.
