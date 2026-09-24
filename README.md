# SEA-RAFT SmartVector for Nuke

This plugin bakes the connected Nuke image stream to a temporary RGB EXR sequence, runs SEA-RAFT in a separate Python process, writes 32-bit float SmartVector EXRs, and reads the cached vector channels back into the source image stream. Inference never runs during normal Nuke frame evaluation.

The worker uses the official [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) `spring-M` configuration and `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` checkpoint. Its Python environment needs the upstream SEA-RAFT requirements plus `OpenEXR` and `numpy`. The checkpoint is downloaded by Hugging Face on first use unless `model` in `job.json` points to a local checkpoint. CUDA is currently required by the Nuke UI.

## Install

1. Clone or install the official SEA-RAFT repository and prepare its CUDA Python environment according to its README.
2. Install `OpenEXR` in that environment, for example `python -m pip install OpenEXR`.
3. Add this repository's `nuke_plugin` directory to Nuke's plugin path in your `~/.nuke/init.py`:

   ```python
   import nuke
   nuke.pluginAddPath(r"E:/EdgeSolver/nuke_plugin")
   ```

4. Restart Nuke. Create **SEA-RAFT → SEA-RAFT SmartVector** from the Nodes menu. Connect the image graph, set **SEA-RAFT Python** and **SEA-RAFT Repository**, choose a cache directory, and click **Generate SmartVectors**.

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

## Current validation boundary

The planning tests run without Nuke or CUDA. A headless Nuke 17 smoke test creates and serializes the Group, merges a supplied reference SmartVector EXR with RGB, and renders a source RGB EXR. The external CUDA worker has not been exercised in this workspace because its SEA-RAFT environment is not available here. Run one short clip through Generate and stock `VectorDistort` in the target environment before production use.
