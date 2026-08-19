# ComfyUI-H3-Multishot-Advance

Focused MiniMax-H3 multi-shot video nodes for ComfyUI.

This project is derived from the MIT-licensed
[ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot)
work by RiftCast / jlucasmcrell. This repository contains ComfyUI custom-node
code. Model files are not bundled; optional example workflows may be included
under `example_workflows/` for reference. The Advance fork keeps the H3
long-form workflow focused: a small node surface, a unified loader, centralized
controls, and a shot cache designed for fast iteration on multi-scene prompts.

## What this node pack is for

MiniMax-H3 is strongest in short clips. This pack chains those clips into a
longer sequence while keeping the workflow practical for prompt iteration:

- write one script with shots separated by `---`;
- change a later shot without re-rendering every earlier compatible shot;
- keep model loading, resolution, seed, frame count, sampler and scheduler in
  one clean workflow lane;
- reuse raw cached shot state instead of rebuilding unchanged clips from final
  MP4/video output.

The main goal is iteration speed without changing the generation path for
uncached shots.

## Nodes

Core nodes:

| Node | Purpose |
| --- | --- |
| `H3MultiLoaderAdvance` | Loads the H3 diffusion model, CLIP/text encoder, video VAE and audio VAE from one node. `clip_type` defaults to `minimax`. |
| `H3ControlsAdvance` | Central render controls: aspect ratio, megapixels, snap multiple, frames per shot, seed, steps, sampler, scheduler and shot count. |
| `H3MultishotMemorySamplerAdvance` | Main multi-shot sampler with continuity options, prompt splitting and shot cache support. |

Support nodes:

| Node | Purpose |
| --- | --- |
| `H3CartridgeLoaderAdvance` | Loads `.riftcast` character/project cartridges when that format is used. |
| `H3SpeedBoostersAdvance` | Applies optional H3 speed/VRAM helper patches when the matching external packs are installed. |

## Prompt format

Use plain text and separate shots with `---`:

```text
Shot 1 prompt...
---
Shot 2 prompt...
---
Shot 3 prompt...
```

JSON is also accepted:

```json
{"prompts": ["Shot 1 prompt...", "Shot 2 prompt..."]}
```

When `shot_count` is higher than the number of prompt blocks, the sampler keeps
the existing fallback behavior and reuses the previous prompt for missing shots.

## Known issues / current workaround

### Silent shots may invent speech unless speech intent is explicit

MiniMax-H3 can sometimes generate random spoken audio in a shot that was meant
to be silent, especially when a previous shot carried a voice reference or
`self_anchor_voice` is enabled. The sampler now guards silent shots by keeping
voice/audio references out of conditioning unless the shot prompt clearly asks
for foreground speech.

Temporary prompt workaround: mark speaking shots explicitly:

```text
[dialogue]
He says: "I finally found it."
```

For shots that should not speak, mark them explicitly as silent:

```text
[silent]
No dialogue. No speech. Ambient sound only.
```

## Shot cache

`shot_cache` defaults to `use_cache`.

The cache is meant for lossless iteration. It stores raw shot state and checks a
stable technical key before reuse. A shot can be reused only when the settings
that affect it still match, including:

- model/checkpoint identity;
- seed and seed-per-shot behavior;
- resolution and frames per shot;
- sampler, scheduler and steps;
- reference image/audio/video fingerprints;
- continuity, pin and bank settings;
- prompt prefix and all previous prompts needed for chain state.

Practical result: if a five-shot workflow is already rendered and only shot 5's
prompt changes, compatible earlier shots can be loaded from cache and the run
continues from the first changed shot. The cache does not improve or degrade
quality by itself; it avoids recomputing matching work.

## Installation

Clone or copy this folder into ComfyUI's `custom_nodes` directory:

```text
ComfyUI/custom_nodes/ComfyUI-H3-Multishot-Advance/
```

Restart ComfyUI and look for:

- `H3 Advance Multi Model Loader`
- `H3 Controls Advance`
- `H3 Advance Multishot Sampler + Memory`

If you use GGUF H3 checkpoints or GGUF text encoders, install
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). This pack includes the
H3 GGUF architecture hook and the `apply_gguf_arch_patch.py` fallback for
installs that need an on-disk patch.

<!-- KursatAs 2026-08-19 14:30: document the conditional context_pin custom-node dependency. -->
### Conditional dependency: Motion Context

The sampler's default `continuity=context_pin` mode needs
[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context).
That pack provides the `MiniMaxH3MotionContext` node used to pin the previous
shot's raw latent tail into the next shot.

If Motion Context is not installed, switch `continuity` to a mode that does not
need it, such as `first_frame`, `cut`, `flf_chain`, or `latent_handoff`.
`ComfyUI-MiniMaxH3-Contex-Loop` is a separate companion/fork and is not a
replacement for the `MiniMaxH3MotionContext` node id expected by `context_pin`.

## Notes

- Model files are not included.
- Example workflows, when present, are reference starting points and may need
  local model paths adjusted.
- This is not a drop-in replacement for old upstream workflows. Node names and
  the workflow layout were intentionally cleaned up.
- The older prompt-writer bundle used by upstream workflows is not bundled here.
  Manual prompt blocks and the sampler's own prompt splitting are the intended
  path.

## Credits

This project builds on work from:

- [ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot)
  by RiftCast / jlucasmcrell.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous and
  contributors.
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) by city96, for optional
  GGUF model support.
- MiniMax-H3 by MiniMax. Model files and their licenses are not included in this
  repository.

The original upstream distribution also referenced
[ComfyUI_JoyAI_Echo_GGUF_Nodes](https://github.com/RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes)
by RealRebelAI for LLM prompt writing. This Advance repository does not bundle
or depend on that package.

## License

MIT. See [LICENSE](LICENSE).

The original MIT copyright notice is preserved, and Advance modifications are
copyright KursatAs.
