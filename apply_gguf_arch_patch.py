"""One-time patch: teach ComfyUI-GGUF the minimax_h3 architecture.

Run from anywhere:  python apply_gguf_arch_patch.py [path-to-ComfyUI-GGUF]
Default path: sibling ComfyUI-GGUF folder in the same custom_nodes dir.
Idempotent - safe to run twice.
"""
import pathlib
import sys

if len(sys.argv) > 1:
    target = pathlib.Path(sys.argv[1]) / "loader.py"
else:
    target = pathlib.Path(__file__).parent.parent / "ComfyUI-GGUF" / "loader.py"

if not target.is_file():
    raise SystemExit(f"ComfyUI-GGUF loader not found at {target} - pass its "
                     f"folder as an argument.")
s = target.read_text(encoding="utf-8")
if "minimax_h3" in s:
    print("already patched.")
    raise SystemExit
old = '"qwen_image"}'
if old not in s:
    raise SystemExit("IMG_ARCH_LIST anchor not found - ComfyUI-GGUF layout "
                     "changed; add \"minimax_h3\" to IMG_ARCH_LIST by hand.")
target.write_text(s.replace(old, '"qwen_image", "minimax_h3"}', 1),
                  encoding="utf-8")
print(f"patched {target} - restart ComfyUI.")
