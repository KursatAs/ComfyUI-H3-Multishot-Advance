# ComfyUI-H3-Multishot-Advance

Focused MiniMax-H3 multi-shot video nodes for ComfyUI.

This project is derived from the MIT-licensed
[ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot)
work by RiftCast / jlucasmcrell. This repository contains ComfyUI custom-node
code. Model files are not bundled; optional example workflows may be included
under `example_workflows/` for reference. The Advance fork keeps the H3
long-form workflow focused: a small node surface, a unified loader, centralized
controls, and a shot cache designed for fast iteration on multi-scene prompts.

<!-- KursatAs 2026-08-19 20:53: surface the project/session resume feature in the README introduction. -->
A key Advance feature is the H3 Advance Project Node. It gives long multi-shot
renders a reusable project/session layer: render a sequence, close ComfyUI,
reopen the same named project later, edit a shot, and continue from the first
changed shot while safe earlier cached state is reused. This is meant for
iterative video work, not for storing final video outputs.

## What this node pack is for

MiniMax-H3 is strongest in short clips. This pack chains those clips into a
longer sequence while keeping the workflow practical for prompt iteration:

- write one script with shots separated by `---`;
- change a later shot without re-rendering every earlier compatible shot;
- save a named project, close ComfyUI, reopen it later and continue editing
  from the first changed shot;
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
| `H3AdvanceProject` / H3 Advance Project Node | Optional project/session node that lets users save a named multi-shot project, close ComfyUI, reopen it later, edit shots, and resume rendering from the first changed shot. |

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

<!-- KursatAs 2026-08-19 20:44: document the reusable H3 Advance Project Node workflow. -->
## H3 Advance Project Node

H3 Advance Project Node is an optional project/session node for long multi-shot H3
workflows. It lets a user save a named project, close ComfyUI, reopen the same
project later, edit one shot, and continue rendering from the first changed shot
instead of starting the whole chain again.

It is not a completed-video archive. Final MP4/video outputs stay in ComfyUI's
normal output folders. The project stores editable render state:

- the effective shot script;
- per-shot prompt hashes;
- sampler/cache compatibility metadata;
- reusable raw prefix cache files.

Project folders are created under:

```text
ComfyUI/user/multishot_advance_projects/<project_name>/
```

### Basic wiring

Add H3 Advance Project Node before `H3 Advance Multishot Sampler + Memory`:

```text
H3 Advance Project Node.project        -> sampler.project
H3 Advance Project Node.script_out     -> sampler.script
H3 Advance Project Node.shot_count_out -> sampler.shot_count
```

Keep the sampler's `shot_cache` set to `use_cache` for normal project reuse.

### Normal use

<!-- KursatAs 2026-08-19 21:00: repeat the close/reopen resume workflow where users read the node usage steps. -->
1. Set a stable `project_name`.
2. Put the shot script into the Project node's `script` input.
3. Use `mode=load_or_create`.
4. Render normally.
5. Later, even after closing and reopening ComfyUI, use the same
   `project_name` to load the saved project, edit the script, and continue from
   the first changed shot.

If shot 2 changes in a three-shot project, the project marks shot 2 as dirty.
The sampler can restore the safe prefix for shot 1, then re-render shot 2 and
shot 3. This is intentional: later shots depend on earlier memory/context state,
so changing shot 2 must invalidate everything after it.

### Project node options

- `shot_count=0` means use the number of shots found in the script. If the
  script already has the exact number of shots, this is the cleanest setting.
- `shot_count>0` makes the Project node produce an effective script with exactly
  that many shots, using the same repeat/truncate behavior as the sampler.
- `mode=load_or_create` is the normal write/update mode.
- `mode=read_only` loads and reports project state without updating
  `project.json` or project cache metadata.
- `mode=rebuild_project` treats the current project as dirty from shot 1, so the
  next project-aware render starts fresh.
- `apply_shot_override` replaces one effective shot prompt before cache
  invalidation is calculated. In write mode, that edited effective script becomes
  the saved project script.

If the Project node's `script` input is empty and a project with the same
`project_name` already exists, the node loads the saved script from
`project.json`. This is the reopen-next-day path.

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

<!-- KursatAs 2026-08-19 21:37: use only the actual MiniMax-H3 fl2va/ref2va checkpoint names in docs. -->
### fl2va/ref2va checkpoint warnings are informational

Some continuity/reference modes can print warnings when a workflow uses an
`fl2va` checkpoint where `ref2va` reference rows would normally be used. These
warnings do not stop the workflow. They tell the user that certain
reference-bank or identity-reference slots may be ignored by that model variant.

This is not always a bad result in practice. "see H3-Multishot-Advance_PROJECT_Example.json"  Many users report that `fl2va` can
produce equal or better visual quality in some scenes, even when the sampler
prints reference-row warnings. If the render continues and the output looks
better for the scene, using `fl2va` is a valid choice. Use `ref2va` when the
workflow specifically depends on reference-row identity behavior.

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

Without H3 Advance Project Node, standalone shot cache files are stored under:

```text
ComfyUI/user/multishot_advance_shot_cache/
```

With H3 Advance Project Node, the reusable prefix cache is stored inside the named
project folder instead, so the project owns its editable render state.

## Installation

Clone or copy this folder into ComfyUI's `custom_nodes` directory:

```text
ComfyUI/custom_nodes/ComfyUI-H3-Multishot-Advance/
```

Restart ComfyUI and look for:

- `H3 Advance Multi Model Loader`
- `H3 Controls Advance`
- `H3 Advance Multishot Sampler + Memory`
- H3 Advance Project Node

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

<!-- KursatAs 2026-08-19 20:51: warn users that project/cache data can grow large on disk. -->
- Important: H3 Advance Project Node folders and shot cache files can become
  large over time. They store raw reusable render state, not just small text
  metadata. Users should periodically clean old projects/cache folders manually
  when they no longer need them.
---
- Model files are not included.
- `example_workflows/` files, when present, are intentionally basic starter
  workflows. They are provided for quick setup and connection reference, not as
  final production graphs. They can be used as starting points with either
  `fl2va` or `ref2va` checkpoints; adjust local model paths, references,
  prompts, and continuity settings for the actual project.
- This is not a drop-in replacement for old upstream workflows. Node names and
  the workflow layout were intentionally cleaned up.
- The older prompt-writer bundle used by upstream workflows is not bundled here.
  Manual prompt blocks and the sampler's own prompt splitting are the intended
  path.

## Credits

This project builds on work from:

- [ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot)
  by RiftCast / jlucasmcrell.
- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
  by NikoDemon80, for the Motion Context node used by `context_pin`.
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
