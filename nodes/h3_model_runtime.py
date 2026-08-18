# -*- coding: utf-8 -*-
"""Model loading and auto activation-reserve runtime for H3 Multishot Advance."""

import json
import re

try:
    from .h3_notify import h3_error as _h3_error
except Exception:
    try:
        from h3_notify import h3_error as _h3_error
    except Exception:
        def _h3_error(*_args, **_kwargs):
            return False


def _h3_fail(message, exc_type=RuntimeError, title="H3 Multishot", tag=None):
    msg = str(message)
    _h3_error(msg, title=title, tag=tag)
    return exc_type(msg)

# ---------------------------------------------------------------------------
# AUTO ACTIVATION RESERVE
#
# VRAM has two tenants with opposite tolerance for being remote. WEIGHTS
# stream well: sequential, known order, prefetched behind ~50s of compute per
# step, so 20GB offloaded costs well under a second. The ALLOCATOR POOL
# (activations) does not stream: it is random-access and re-touched all step,
# and when it does not fit, the driver evicts blind mid-step. Measured on a
# 3090: starving the pool by ~3GB = 533 s/it vs 99 s/it. Measured on a 5090:
# 168W at "99% utilisation" - a card waiting, not computing.
#
# So the correct split is reserve >= pool, and stream whatever weights do not
# fit - overshooting is nearly free, undershooting is 5-10x. The pool scales
# with the render shape, which is why any hand-set number (a GB figure, a
# memory_usage_factor) is correct for exactly one resolution and a trap at
# every other: LOWER the resolution with a fixed factor and the reserve
# shrinks below the pool - the render gets SLOWER, the opposite of what any
# person expects.
#
# This engine removes the knob:
#   - memory_required(input_shape) is overridden with a function of the
#     ACTUAL shape comfy passes at load time - never a constant.
#   - Unmeasured shapes reserve 60% of currently-free VRAM: generous enough
#     to be cliff-proof at any resolution, and only slightly slower than
#     optimal (a few more GB of weights stream).
#   - Our samplers measure the true allocator peak of every run and cache it
#     per (GPU, model, shape-cells) in the user dir. From the second run at
#     a shape, the reserve is measured * 1.25 - per machine, no telemetry to
#     read, no number to know.
# ---------------------------------------------------------------------------

_AUTO_FLOOR = 8 * 1024**3          # never reserve less: workspaces + margin
# First-run fraction of free VRAM. Deliberately HIGH: over-reserving merely
# streams more weights (<1s/step behind 50-100s of compute), while
# under-reserving is the 5-10x cliff. 0.60 was calibrated on a 32GB card and
# proved WRONG on a 24GB one: 60% of the 3090's 21.9GB free = 13.2GB against
# a ~17.5GB pool -> max_reserved 25.06GB on a 24GB card -> 492 s/it. At 0.88
# the same card reserves 19.3GB and streams the difference. Measurement then
# tightens DOWN from the safe side.
_AUTO_FRACTION = 0.88              # unmeasured shapes: fraction of free VRAM
_AUTO_WEIGHT_NUCLEUS = 2 * 1024**3  # always leave a little room for weights
_AUTO_MARGIN = 1.25                # measured pool -> reserve headroom
_AUTO_KEEPOUT = 1024 * 1024**2     # left free beyond weights+pool. 384 MB was
                                   # too small: ComfyUI reserves its own ~117 MB
                                   # buffer and counts 'usable' differently from
                                   # get_free_memory, so a clamp computed to keep
                                   # 20.3 GB of weights resident still offloaded
                                   # 401 MB and then aborted in a CUDA kernel.
_AUTO_MIN_POOL = 1536 * 1024**2    # below this, prefer a loud OOM to a silent crawl
_auto_cache = None                 # lazy {key: pool_bytes}
_auto_last = {"key": None, "model": None}   # what the next sampling run is
_auto_session = {}                 # key -> reserve pinned for this session


def _auto_cache_path():
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import os
    return os.path.join(base, "h3_auto_reserve.json")


_AUTO_SCHEMA = 2  # bumped when the pool measurement changed meaning


def _auto_cache_load():
    global _auto_cache
    if _auto_cache is None:
        import io as _io, os
        _auto_cache = {}
        p = _auto_cache_path()
        if os.path.isfile(p):
            try:
                _auto_cache = json.load(_io.open(p, encoding="utf-8"))
            except Exception:
                _auto_cache = {}
        # Entries written before the full-weights fix can be too large by
        # however much the DiT had been offloaded that shot, and the store
        # keeps the largest value forever - so one bad shot poisons a shape
        # permanently. Drop pre-schema caches once instead of carrying them.
        if _auto_cache.pop("_schema", None) != _AUTO_SCHEMA:
            _stale = len(_auto_cache)
            if _stale:
                print("[H3AutoReserve] discarded %d cached reserve(s) written "
                      "before the partial-load measurement fix. Those numbers "
                      "could be inflated by the offloaded weight bytes and "
                      "never shrank. They re-measure on the next run."
                      % _stale, flush=True)
            _auto_cache = {}
    return _auto_cache


def _auto_cache_store(key, pool_bytes):
    """Record a pool measurement. Returns True if it raised the stored value."""
    cache = _auto_cache_load()
    prev = cache.get(key, 0)
    # keep the largest pool ever seen for the shape; shrinking on a lucky
    # run risks the cliff on the next unlucky one
    if pool_bytes <= prev:
        return False
    cache[key] = int(pool_bytes)
    try:
        import io as _io
        _out = dict(cache)
        _out["_schema"] = _AUTO_SCHEMA
        _io.open(_auto_cache_path(), "w", encoding="utf-8").write(
            json.dumps(_out, indent=1))
    except Exception as e:
        print(f"[H3AutoReserve] cache write failed ({e}) - measurements "
              f"will not persist across restarts", flush=True)
    return True


def _auto_cache_floor(key, pool_bytes):
    """Record a LOWER BOUND from a run whose peak was truncated by the card.

    Same ratchet as _auto_cache_store - it only ever raises - but named
    separately at the call site so it is obvious that the number is known to
    understate the real need. Without this a ceiling-bound shot contributes
    nothing at all, which is how the chained shots deadlocked: their payload key
    stayed empty, the fallback reserve was too small for them, and every attempt
    to measure was thrown away for being too small.
    """
    return _auto_cache_store(key, pool_bytes)


_auto_payload = {"sig": ""}


def _auto_set_payload(sig):
    """Samplers call this before each shot with a conditioning-payload
    signature (keyframes / reference blocks / audio refs). Two shots with
    the same latent shape but different payloads need DIFFERENT pools -
    render-verified 2026-08-10: shot 1 (bare) measured a 4.9 GB pool,
    shot 2 (chain keyframe + 10s self-anchor audio ref) then spilled to
    system RAM at that reserve and ran 5x slower per step."""
    _auto_payload["sig"] = str(sig or "")


def _payload_scheme(sig):
    """Which sampler wrote a payload signature: legacy non-memory sampler
    signatures start with "kf"; memory-sampler signatures use continuity names. The two
    namespaces never compare equal, so a raw string mismatch is NOT evidence
    of a heavier payload (24 GB test-lab finding F008, 2026-08-16)."""
    return "kf" if str(sig or "").startswith("kf") else "cont"


def _payload_mult(src_sig, dst_sig):
    """Bare->payload scale when borrowing across signatures. Same string or
    different sampler scheme -> 1.0 (equal-cells evidence taken as-is);
    within one scheme the measured x1.6 bare->payload jump stands. The old
    flat "1.6 unless identical" fired on every CORE run against a
    Memory-sampler cache: 12.4 GB became a 19.8 GB request, 129 MB of DiT
    stayed resident, 66 s/it (24 GB lab, S1)."""
    if src_sig == dst_sig:
        return 1.0
    if _payload_scheme(src_sig) != _payload_scheme(dst_sig):
        return 1.0
    return 1.6


_PAYLOAD_ADD_BYTES = 4 * 1024**3    # a keyframe + an audio ref, in GB, roughly fixed


def _payload_need(bare, src_sig, dst_sig):
    """Pool NEED for a payload signature, from a bare (or other) measurement.

    24 GB test-lab finding F010 (2026-08-16/17): the flat x1.6 was above the
    93rd percentile of 16 measured bare->payload pairs (median 1.40x, max
    1.73x) and the cost is ADDITIVE, not multiplicative - a largely fixed
    number of GB, so a big relative jump on a 6 GB pool and negligible on a
    15 GB one. Measured: 6 GB bare -> ~1.5x; 11 GB -> +3.1..+4.0 GB;
    15 GB -> ~1.1x. A flat multiplier therefore over-reserved worst at large
    geometry - exactly where a 24 GB card evicts weights instead (S3 run 3
    reserved 20.6 GB for a 15.4 GB need). Model it as +4 GB capped at the old
    x1.6, so it can only ask for less than before, never more.
    """
    m = _payload_mult(src_sig, dst_sig)
    if m <= 1.0:
        return int(bare)
    return int(min(bare * 1.6, bare + _PAYLOAD_ADD_BYTES))


def _model_family(stem):
    """Coarse quant family for the reserve borrow. The pool tracks geometry
    WITHIN a family, not across: comfy-native quant paths (w4a8/int8/nvfp4)
    carry dequant scratch that GGUF does not - the same card's cache rates
    read 3.0-5.7 GB/Mcell for w4a8 against 2.2-2.5 for GGUF, and a GGUF-
    based borrow under-reserved a w4a8 first run into WDDM paging (the 24 GB lab box lab
    F009, 2026-08-16). Under-reserving is the fatal direction."""
    s = str(stem or "").lower()
    if re.search(r"q\d|gguf|curve|k_[msl]|_q[0-9]", s):
        return "gguf"
    return "native"


_CROSS_FAMILY_MULT = 1.8   # w4a8/gguf per-cell ratio measured on the 3090 cache


def _auto_key(model_name, cells, sig=None):
    try:
        import torch
        dev = torch.cuda.get_device_name(0)
    except Exception:
        dev = "cpu"
    import os
    stem = os.path.splitext(os.path.basename(model_name))[0]
    s = _auto_payload["sig"] if sig is None else sig
    return f"{dev}|{stem}|{cells}|{s}"


def _install_auto_reserve(patcher, model_name):
    """Shape-aware memory_required on the BaseModel (clone-safe)."""

    def memory_required(input_shape, *args, **kwargs):
        cells = 1
        try:
            for d in list(input_shape)[1:]:
                cells *= int(d)
        except Exception:
            cells = 0
        key = _auto_key(model_name, cells)
        _auto_last["key"] = key
        # comfy (and DynamicVRAM) call memory_required repeatedly - per load
        # AND per sampling step. The answer must be STABLE for a shape:
        # recomputing "60% of free" as free shrinks is a feedback loop
        # (reserving memory reduces free, which reduces the next answer).
        # Pin the first computation per (model, shape) for the session;
        # a fresh measurement invalidates the pin.
        pinned = _auto_session.get(key)
        if pinned is not None:
            return pinned
        cache = _auto_cache_load()
        measured = cache.get(key) or 0
        _known_need = int(measured)   # best evidence of true pool need
        if measured:
            # a real measurement carries its own x1.25 margin - the old
            # 8 GB floor here overrode good small measurements and forced
            # weight offload on 24 GB cards for nothing
            reserve = max(int(measured * _AUTO_MARGIN), 2 * 1024**3)
            how = f"measured pool {measured/2**30:.1f} GB x {_AUTO_MARGIN}"
        else:
            # unmeasured payload variant of a measured shape: estimate from
            # the sibling instead of guessing from free VRAM. Reference and
            # keyframe tokens ride every step; x1.6 covered the measured
            # bare->payload jump with margin.
            sib = sib_sig = None
            prefix = key.rsplit("|", 1)[0] + "|"
            mysig = key.rsplit("|", 1)[1]
            for k2, v2 in cache.items():
                if k2.startswith(prefix) and v2 and (sib is None or v2 > sib):
                    sib, sib_sig = v2, k2.rsplit("|", 1)[1]
            if sib:
                _known_need = _payload_need(sib, sib_sig, mysig)
                reserve = max(int(_known_need * _AUTO_MARGIN), _AUTO_FLOOR)
                how = (f"payload variant of a measured shape: sibling pool "
                       f"{sib/2**30:.1f} GB -> need {_known_need/2**30:.1f} GB "
                       f"(payload +4 GB capped x1.6) x {_AUTO_MARGIN}")
            else:
                # FIRST RUN at an unseen (model, shape). The pool tracks the
                # GEOMETRY, not the checkpoint - measured 2026-08-16: a Q8
                # GGUF and a mixed-precision file wanted the same ~11-13 GB
                # at the same cells. So borrow the nearest measurement on
                # this card from ANY model and shape, scaled by cell count,
                # before falling back to the free-VRAM placeholder - which
                # planned 3.3 GB for an 11 GB pool twice in one afternoon
                # and produced a crawl on a 5090 and a dead CUDA context on
                # a 3090. Max over candidates: over-reserving streams a few
                # GB of weights (cheap); under-reserving pages (fatal).
                borrowed = bfrom = None
                try:
                    mydev, _, _, mysig = key.split("|", 3)
                    # same-payload siblings first; only if none exist, other
                    # payloads with the bare->payload x1.6 on top. Otherwise a
                    # bare shot 1 borrows a pinned shot's pool times 1.6 and
                    # over-reserves by double.
                    # nearest shape wins, NOT the largest estimate: pools are
                    # not purely linear in cells (fixed overhead fattens the
                    # per-cell rate of small shapes), so extrapolating far
                    # overshoots - a real cache here scaled small w4a8 shapes
                    # to 26 GB while two same-cells entries said 14-15.
                    # rank: shape distance first, then same payload, then the
                    # larger estimate. A same-cells measurement beats a
                    # same-payload one from a distant shape - a real cache
                    # extrapolated distant small shapes to 26-31 GB while
                    # same-cells entries said 15.
                    best = None
                    myfam = _model_family(key.split("|", 3)[1])
                    for k2, v2 in cache.items():
                        try:
                            d2, m2, c2, s2 = k2.split("|", 3)
                            c2 = int(c2)
                        except ValueError:
                            continue
                        if d2 != mydev or not v2 or not c2 or not cells:
                            continue
                        ratio = cells / c2
                        if not (0.4 <= ratio <= 2.5):
                            continue   # no wild-scale extrapolation
                        fam_miss = 0 if _model_family(m2) == myfam else 1
                        est = int(_payload_need(v2 * ratio, s2, mysig)
                                  * (_CROSS_FAMILY_MULT if fam_miss else 1.0))
                        # same quant family first (F009), then nearest shape,
                        # then same payload, then the fatter estimate
                        rank = (fam_miss, abs(ratio - 1.0),
                                0 if s2 == mysig else 1, -est)
                        if best is None or rank < best:
                            best, borrowed, bfrom = rank, est, (c2, v2)
                except Exception:
                    borrowed = None
                if borrowed:
                    _known_need = borrowed
                    reserve = max(int(borrowed * _AUTO_MARGIN), _AUTO_FLOOR)
                    how = ("borrowed pool: %.1f GB measured at cells=%d "
                           "scaled to %.1f GB (pool tracks geometry, not "
                           "the checkpoint)"
                           % (bfrom[1] / 2**30, bfrom[0], borrowed / 2**30))
                    print("[H3AutoReserve] first run with this model at this "
                          "shape - borrowing a measured pool from another "
                          "model/shape on this card: %.1f GB at cells=%d "
                          "scales to %.1f GB here. This shot still records "
                          "its own measurement."
                          % (bfrom[1] / 2**30, bfrom[0], borrowed / 2**30),
                          flush=True)
                else:
                    try:
                        import comfy.model_management as mm
                        free = mm.get_free_memory(mm.get_torch_device())
                    except Exception:
                        free = 24 * 1024**3
                    reserve = max(int(min(free * _AUTO_FRACTION,
                                          free - _AUTO_WEIGHT_NUCLEUS)),
                                  _AUTO_FLOOR)
                    how = (f"first run at this shape: "
                           f"{_AUTO_FRACTION:.0%} of free")
        # CLAMP AGAINST THE CARD. Every GB reserved here comes out of the
        # weights budget, and a DiT that misses a FULL load streams the
        # remainder over PCIe every step. Measured 2026-08-12 at 960x544:
        # shot 1 reserved 7.8 GB and loaded completely at 18.8 s/it; shot 2's
        # larger payload reserved 9.4 GB, left the DiT 399 MB short, and ran
        # at 283 s/it - a 15x collapse bought by 1.6 GB of headroom the
        # measurement said was not needed. The failure modes are asymmetric:
        # too small OOMs loudly and you fix it, too large silently costs 15x.
        # So the pool yields to the weights - but ONLY while the cut stays at
        # or above the measured need. Cutting below it does not keep the
        # weights resident (the allocator evicts them for real allocations
        # regardless - measured on a 3090: clamped to 1.5 GB "to keep weights
        # resident" and the shot's own measurement then read resident 0.0),
        # so past that line the weights yield instead.
        try:
            import comfy.model_management as _cm
            _dev = _cm.get_torch_device()
            _free = _cm.get_free_memory(_dev)
            _w = int(patcher.model_size())
            _cap = int(_free - _w - _AUTO_KEEPOUT)
            if _cap <= 0:
                # The weights alone exceed free VRAM. Reserving the measured
                # pool here is the WORST possible move: every byte of reserve
                # pushes another byte of weights out, and the old `_cap > 0`
                # guard skipped the clamp entirely in exactly this case. A
                # 3090 with 10.3 GB free and 20.3 GB of weights reserved
                # 22.1 GB and loaded "0.00 MB usable, 0.00 MB loaded,
                # 20796.43 MB offloaded" - every layer streamed off disk on
                # every step, 128 s/it against ~29 s/it resident, and after
                # four hours the file-reader gave out with
                # hostbuf_file_reader_read failed.
                _was = reserve
                # NOT the floor. Reserving the minimum guarantees the weights
                # load but starves the activation pool, and the allocator then
                # spills ACTIVATIONS instead - measured worse than the problem
                # it replaced: 36 -> 55 -> 331 s/it across three shots while
                # every shot still reported 'loaded completely, full load:
                # True'. Use the measured pool when a sibling shot has given
                # us one: shot 2 measured 8.1 GB and peaked at 22.6 with 14.5
                # GB of weights, which fits a 24 GB card exactly. The inflated
                # payload estimate (x1.6 x1.25 compounding off an already
                # bumped sibling) is what asks for 16 GB and does not fit.
                reserve = (max(int(_known_need), _AUTO_MIN_POOL)
                           if _known_need else _AUTO_MIN_POOL)
                reserve = max(min(reserve, int(_free - _AUTO_KEEPOUT)),
                              _AUTO_MIN_POOL)
                how += (" | TIGHT %.1f -> %.1f GB: weights %.1f GB vs "
                        "%.1f GB reported free"
                        % (_was / 2**30, reserve / 2**30, _w / 2**30,
                           _free / 2**30))
                print("[H3AutoReserve] TIGHT: %.1f GB of weights against "
                      "%.1f GB reported free, so this shot has no headroom. "
                      "Reserving %.1f GB (the measured pool) rather than the "
                      "payload estimate, which does not fit. Note the reported "
                      "figure understates what ComfyUI ends up with, so the "
                      "weights may still load completely - watch for a spill "
                      "instead: high GPU utilisation at low wattage. If the "
                      "render crawls, lower frames_per_shot or resolution, free "
                      "the other ComfyUI instance, or load a smaller DiT."
                      % (_w / 2**30, _free / 2**30, reserve / 2**30), flush=True)
                # Driver headroom applies HERE too. Reported free is misleading
                # in this branch (resident weights count as used but get
                # reused), so the peak still lands wherever weights+pool put
                # it - measured 2026-08-16 on a 5090: the tight requeue of a
                # shape whose first run peaked at a healthy 29.4/32.6 sat at
                # 31.5/32.6 and crawled at 142 W. The main headroom block below
                # is gated on _cap > 0 and its clamp uses reported free, both
                # wrong for this regime, so bump against TOTAL directly:
                # streaming ~2 GB more weights is cheap, the last few percent
                # of VRAM are not.
                try:
                    _total_t = _cm.get_total_memory(_dev)
                except Exception:
                    _total_t = _free
                _bump = max(0, int(_total_t * 0.09) - _AUTO_KEEPOUT)
                if _bump:
                    reserve += _bump
                    how += (" | +driver headroom (tight) +%.1f GB"
                            % (_bump / 2**30))
                    print("[H3AutoReserve] driver headroom (tight path): "
                          "raising the reserve by %.1f GB so extra weights "
                          "stream instead of the peak riding the last few "
                          "percent of VRAM (that zone measured 2-12x slower)."
                          % (_bump / 2**30), flush=True)
            elif reserve > _cap:
                _was = reserve
                _card_max = max(int(_free - _AUTO_KEEPOUT), _AUTO_MIN_POOL)
                if _known_need and _cap < _known_need:
                    # The cut would land below the measured need. Weight
                    # residency is not achievable in this regime - the
                    # allocator evicts weights to satisfy the sampler's real
                    # allocations no matter what is reserved (3090: clamped to
                    # 1.5 GB "to keep weights resident", measurement then read
                    # resident 0.0, 24.3 GB driver spill, ~2x step time). So
                    # give the pool its need, bounded by the card, and let the
                    # weights stream: that is the cheaper side here.
                    # Bare need, not need*margin. The x1.25 margin guards a
                    # pool overrun against PINNED weights; here the weights
                    # stream regardless, so an overrun just evicts a little
                    # more of them - graceful. Every GB of margin trimmed is a
                    # GB of weights that stays resident instead of streaming
                    # every step (measured: 18.4 reserve left 3.3 GB resident,
                    # 14.7 leaves 7.0).
                    reserve = min(max(int(_known_need), _AUTO_MIN_POOL),
                                  _card_max)
                    _resid = max(0, int(_free - _AUTO_KEEPOUT) - reserve)
                    how += (" | NEED %.1f -> %.1f GB (bare need, margin "
                            "yielded to weights): ~%.1f GB of weights can stay "
                            "resident"
                            % (_was / 2**30, reserve / 2**30, _resid / 2**30))
                    print("[H3AutoReserve] pool need %.1f GB cannot fit "
                          "beside %.1f GB of weights in %.1f GB free. "
                          "Reserving %.1f GB for the pool and letting the "
                          "weights stream - clamping the pool here does not "
                          "keep the weights resident, it only adds a driver "
                          "spill on top of the offload."
                          % (_known_need / 2**30, _w / 2**30,
                             _free / 2**30, reserve / 2**30), flush=True)
                    if _card_max < _known_need:
                        print("[H3AutoReserve] WARNING: even with zero weights "
                              "resident the card has %.1f GB for a %.1f GB "
                              "pool. This can die inside a CUDA kernel. Lower "
                              "frames_per_shot or resolution."
                              % (_card_max / 2**30, _known_need / 2**30),
                              flush=True)
                else:
                    # No measurement says the cut goes below need, so this is
                    # margin-trimming: keep the weights resident. Measured
                    # 2026-08-12: a 399 MB weight shortfall cost 15x.
                    reserve = max(_cap, _AUTO_MIN_POOL)
                    how += (" | CLAMPED %.1f -> %.1f GB to keep the weights "
                            "(%.1f GB) resident out of %.1f GB free"
                            % (_was / 2**30, reserve / 2**30, _w / 2**30,
                               _free / 2**30))
                    # Only shout when the pre-clamp figure came from EVIDENCE.
                    # On a first run at an unseen shape there is no
                    # measurement, and `_was` is the placeholder from the
                    # "%.0f%% of free" branch - a fraction of whatever happens
                    # to be free, not an estimate of this shot's need. It is
                    # therefore always far above the clamp, so this warning
                    # fired on every new shape and predicted a server-killing
                    # crash for renders that were fine.
                    #
                    # Field log 2026-08-15, 5090, two different shapes:
                    #   cells=6384960  "asked for 26.5 GB"
                    #   cells=6993216  "asked for 26.5 GB"   <- 10% more pixels
                    # Identical, because both are 88% of the same 30.1 GB free.
                    # Both clamped, both rendered clean, and both then MEASURED
                    # 11.4 GB - less than half the figure being warned about.
                    # A warning that cries wolf on every new resolution teaches
                    # people to cancel jobs that would have worked.
                    if _was > reserve * 1.35 and _known_need:
                        print("[H3AutoReserve] WARNING: this shape previously "
                              "MEASURED a %.1f GB activation pool and only "
                              "%.1f GB is available after the %.1f GB of "
                              "weights. That is not a slow render - it usually "
                              "dies inside a CUDA kernel and takes the server "
                              "with it. Lower frames_per_shot or resolution, or "
                              "load a smaller quantisation of the DiT."
                              % (_known_need / 2**30, reserve / 2**30,
                                 _w / 2**30), flush=True)
                    elif _was > reserve * 1.35:
                        print("[H3AutoReserve] first run at this shape - the "
                              "%.1f GB figure is a placeholder (%.0f%% of "
                              "free), not a measurement, and has been clamped "
                              "to %.1f GB so the %.1f GB of weights stay "
                              "resident. The real pool gets measured during "
                              "this shot and used from the next one on."
                              % (_was / 2**30, _AUTO_FRACTION * 100,
                                 reserve / 2**30, _w / 2**30), flush=True)
                if measured and reserve < measured:
                    how += (" [tight: below the measured pool %.1f GB]"
                            % (measured / 2**30))
            # DRIVER HEADROOM (measured 2026-08-16). Five identical runs
            # at ~96% VRAM took 27-175 minutes - same shape, same seed, a
            # lottery. The same run with the reserve raised so ~3.6 GB of
            # weights streamed instead peaked at 29.5/32.6 and took 15
            # minutes, faster than every resident run. WDDM demotes
            # unpredictably in the last few percent of VRAM, and streamed
            # weights are cheap (core prefetch overlaps them) - so when
            # weights + pool would land in that zone, RAISE the reserve:
            # it is the one lever that pushes weights off and peak down.
            if _cap > 0:
                try:
                    _total = _cm.get_total_memory(_dev)
                except Exception:
                    _total = _free
                _wddm = int(_total * 0.09)
                _pool_real = int(_known_need) if _known_need else reserve
                _target = _pool_real + max(0, _wddm - _AUTO_KEEPOUT)
                if (_w + _pool_real + _AUTO_KEEPOUT + _wddm > _free
                        and reserve < _target):
                    _target = min(_target,
                                  max(int(_free - _AUTO_KEEPOUT),
                                      _AUTO_MIN_POOL))
                    how += (" | +driver headroom %.1f -> %.1f GB"
                            % (reserve / 2**30, _target / 2**30))
                    print("[H3AutoReserve] driver headroom: raising the "
                          "reserve %.1f -> %.1f GB so ~%.1f GB of weights "
                          "stream instead of riding the last few percent "
                          "of VRAM (that zone measured 2-12x slower)."
                          % (reserve / 2**30, _target / 2**30,
                             max(0.0, (_w + _pool_real + _AUTO_KEEPOUT
                                       + _wddm - _free)) / 2**30),
                          flush=True)
                    reserve = _target
        except Exception:
            pass
        _auto_session[key] = reserve
        print(f"[H3AutoReserve] shape cells={cells}: reserving "
              f"{reserve/2**30:.1f} GB ({how})", flush=True)
        return reserve

    patcher.model.memory_required = memory_required
    patcher.memory_required = memory_required
    _auto_last["model"] = model_name


def _auto_measure_begin():
    """Call right before sampling: snapshot the allocator + clock."""
    import time as _t
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        return {"res": torch.cuda.memory_reserved(), "t0": _t.time()}
    except Exception:
        return {"res": None, "t0": _t.time()}


def _auto_measure_end(before, patcher=None, steps=None):
    """Call right after sampling: cache the real pool, and DETECT the two
    silent failure modes by name - a system-RAM spill (peak at the card's
    ceiling + step time collapsed) used to present as an unexplained 5-10x
    slowdown with nothing in the log."""
    import time as _t
    if not isinstance(before, dict):
        before = {"res": before, "t0": None}
    key = _auto_last["key"]
    if key is None:
        return
    # ---- step-time tracking (works even when CUDA stats are unavailable)
    sit = None
    if before.get("t0") and steps:
        sit = (_t.time() - before["t0"]) / max(1, int(steps))
        base_key = "sit|" + key.rsplit("|", 1)[0]   # same shape, any payload
        best = _auto_session.get(base_key)
        if best is None or sit < best:
            _auto_session[base_key] = sit
        elif best > 0 and sit > 2.5 * best:
            print(f"[H3AutoReserve] SLOWDOWN: {sit:.0f}s/step vs "
                  f"{best:.0f}s/step earlier this session ({sit/best:.1f}x). "
                  f"This is the VRAM-spill signature: the driver is paging "
                  f"to system RAM instead of erroring. Fix: raise the "
                  f"activation reserve, drop resolution/frames, or remove "
                  f"reference payload (audio refs / keyframes ride every "
                  f"step).", flush=True)
    if before.get("res") is None:
        return
    try:
        import torch
        peak = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        if peak >= total * 0.97:
            print(f"[H3AutoReserve] WARNING: peak reserved "
                  f"{peak/2**30:.1f} GB of {total/2**30:.1f} GB - the "
                  f"allocator hit the card's ceiling; any overflow was "
                  f"paged to system RAM by the driver (silent, slow).",
                  flush=True)
        loaded = 0
        try:
            loaded = int(patcher.loaded_size()) if patcher is not None else 0
        except Exception:
            try:
                loaded = int(getattr(patcher.model,
                                     "model_loaded_weight_memory", 0))
            except Exception:
                loaded = 0
        full = 0
        try:
            full = int(patcher.model_size()) if patcher is not None else 0
        except Exception:
            full = 0
        # `peak` is max_memory_reserved() - VRAM. Weights that were offloaded
        # to the host were never in VRAM and so were never in peak. Subtract
        # the RESIDENT bytes, not the full model size.
        #
        # Whether the sample is usable depends on residency, in three cases:
        #
        #   fully offloaded  peak IS the pool, nothing to subtract. This is the
        #                    cleanest sample available - record it. On a 24 GB
        #                    card at production shapes this is the NORMAL mode,
        #                    not a fault, so discarding it stops all learning.
        #   partial          peak is bounded by what fit on the card rather than
        #                    by what the shot wanted. Measures the ceiling, not
        #                    the need. Discard.
        #   fully resident   subtract the weights and record, as always.
        #
        # An earlier version of this collapsed the first two cases and discarded
        # both, which wiped the cache and then left every shot reporting "first
        # run at this shape" - 1.5 GB reserves, 24.3 GB of driver spill and ~2x
        # step time on the 3090. Diagnosed there with before/after logs.
        pool = peak - before["res"] - loaded
        _frac = (loaded / float(full)) if full else 1.0
        _partial = bool(full) and 0.02 < _frac < 0.98
        # 2.2.5: a blanket discard on every partial load DEADLOCKS the chained
        # shots. Field log 2026-08-14, 5090, Q8_0 at 736x1280x243: shot 1 loads
        # completely and measures fine, shots 2-4 carry the context pin plus the
        # self-anchor audio ref, cannot fit the 20.8 GB DiT beside them, load
        # partially - and are then discarded. Their payload key therefore never
        # receives a single measurement, so it keeps the fallback reserve, so it
        # keeps loading partially. Learning requires a full load; a full load
        # requires the knowledge the rule refuses to record. Every shot after the
        # first ran 57 s/it against shot 1's 39.
        #
        # The discard is only actually justified when the peak was CEILING-BOUND
        # - i.e. the run pressed against the card and the pool got truncated. If
        # peak sits well below total VRAM the activations completed normally and
        # the pool figure is honest, even though some weights were streaming. So
        # gate the discard on proximity to the ceiling instead of on residency
        # alone, and when we do discard, still keep the observation as a FLOOR so
        # the estimate can only ratchet toward the truth rather than never move.
        _total = 0
        try:
            import torch as _t
            _total = int(_t.cuda.get_device_properties(0).total_memory)
        except Exception:
            _total = 0
        _ceiling_bound = bool(_total) and peak > _total * 0.94
        if _partial and _ceiling_bound:
            _floor = _auto_cache_floor(key, pool)
            print(f"[H3AutoReserve] measurement capped: only "
                  f"{loaded/2**30:.1f} of {full/2**30:.1f} GB of weights were "
                  f"resident AND peak {peak/2**30:.1f} GB pressed the "
                  f"{_total/2**30:.1f} GB card, so the pool was truncated. "
                  f"Keeping {pool/2**30:.1f} GB as a floor"
                  f"{' (raised)' if _floor else ''}; not treating it as the "
                  f"full need.", flush=True)
        elif _partial:
            # streaming weights, but the pool itself was never squeezed
            if pool > 512 * 1024**2:
                _auto_cache_store(key, pool)
                _auto_session.pop(key, None)
                print(f"[H3AutoReserve] measured pool {pool/2**30:.1f} GB with "
                      f"weights streaming ({loaded/2**30:.1f} of "
                      f"{full/2**30:.1f} GB resident). Peak {peak/2**30:.1f} GB "
                      f"stayed clear of the {_total/2**30:.1f} GB ceiling, so "
                      f"the activation figure is sound - recording it. This is "
                      f"the sample chained shots could never contribute before.",
                      flush=True)
        elif bool(full) and _frac <= 0.02 and pool > 512 * 1024**2:
            # ~0% of the weights were resident: the run streamed everything,
            # and with the whole card to itself the pool inflates to fill
            # whatever reserve it was handed (allocator caches). If the
            # figure hugs the reserve it is an artifact OF the reserve, and
            # storing it locks the shape into all-streaming forever -
            # measured 2026-08-16 on the 3090: a cancelled 21.8 GB-reserve
            # run recorded a "21.7 GB pool" at a shape two healthy runs had
            # measured at 14-15.
            _pinned = _auto_session.get(key) or 0
            if _pinned and pool >= _pinned * 0.85:
                print(f"[H3AutoReserve] discarding this shot's pool figure "
                      f"({pool/2**30:.1f} GB): no weights were resident and "
                      f"it hugs the {_pinned/2**30:.1f} GB reserve, so it "
                      f"measures the reserve, not the need.", flush=True)
            else:
                _auto_cache_store(key, pool)
                _auto_session.pop(key, None)
                print(f"[H3AutoReserve] measured pool {pool/2**30:.1f} GB "
                      f"with all weights streaming - it sits well under the "
                      f"reserve, so the figure is real need.", flush=True)
        elif pool > 512 * 1024**2:          # ignore no-op runs
            _auto_cache_store(key, pool)
            _auto_session.pop(key, None)     # re-pin from measured
            print(f"[H3AutoReserve] measured pool {pool/2**30:.1f} GB for "
                  f"this shape+payload (peak {peak/2**30:.1f} - resident "
                  f"weights {loaded/2**30:.1f}) - next run reserves "
                  f"{max(pool*_AUTO_MARGIN, 2*1024**3)/2**30:.1f} GB",
                  flush=True)
    except Exception:
        pass


class _H3ModelLoaderSupport:
    """Internal model-loading support used by H3MultiLoader."""

    @staticmethod
    def _list_names(folder="diffusion_models"):
        """Every model file the dropdown offers: core's list plus a RECURSIVE
        walk for .gguf, which is not in supported_pt_extensions so
        get_filename_list never returns it (and a flat listdir misses
        anything filed under diffusion_models/gguf/). Shared by INPUT_TYPES
        and _resolve_name - the resolver used to consult get_filename_list
        alone, so its "moved into a gguf/ subfolder" fallback could never
        find a gguf (2.6.0, found on the 24 GB box)."""
        import folder_paths
        import os
        try:
            files = list(folder_paths.get_filename_list(folder))
        except Exception:
            files = []
        gguf = []
        try:
            dirs = folder_paths.get_folder_paths(folder)
        except Exception:
            dirs = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, _dirs, fs in os.walk(d):
                for f in fs:
                    if f.lower().endswith(".gguf"):
                        gguf.append(os.path.relpath(os.path.join(root, f), d))
        return sorted(set(files) | set(gguf))

    @classmethod
    def input_types(cls):
        names = cls._list_names("diffusion_models")
        return {"required": {"model_name": (names, {
            "tooltip": "safetensors or GGUF - loader routes automatically."})},
            "optional": {"activation_reserve_gb": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                "tooltip": "0 = AUTO (recommended). The pack sizes the "
                "activation reserve for the actual render shape, measures the "
                "real peak each run, and tightens itself per machine - lower "
                "resolutions get faster automatically. Set a number only to "
                "pin the reserve by hand; that number is for ONE resolution "
                "and the wrong number is 5-10x slower, not a little slower."})}}

    @staticmethod
    def _resolve_name(model_name):
        """F007 (24 GB lab): a saved combo value stops matching the moment the
        model tree is reorganised (a file moved into a gguf/ subfolder, or a
        workflow saved on a box that keeps it at the root). Rather than fail
        the whole queue with value_not_in_list, fall back to a UNIQUE basename
        match across the folders this loader reads, and say what was resolved.
        Ambiguous (two files, same basename) stays an error - guessing wrong
        is worse than stopping."""
        import folder_paths
        import os
        want = os.path.basename(str(model_name)).lower()
        cands = []
        for folder in ("diffusion_models", "unet", "checkpoints"):
            try:
                for f in _H3ModelLoaderSupport._list_names(folder):
                    if os.path.basename(f).lower() == want:
                        cands.append((folder, f))
            except Exception:
                pass
        # exact match wins silently
        for folder, f in cands:
            if f == model_name:
                return model_name
        uniq = sorted(set(f for _, f in cands))
        if len(uniq) == 1 and uniq[0] != model_name:
            print("[H3ModelLoader] %r is not at the saved path; resolved by "
                  "basename to %r (the model tree moved since this workflow "
                  "was saved)." % (model_name, uniq[0]), flush=True)
            return uniq[0]
        return model_name

    @staticmethod
    def _warn_quant_backend(model_name):
        """24 GB lab finding, 2026-08-17: on torch < cu130 comfy-kitchen's cuda
        backend is disabled and triton is off unless --enable-triton-backend, so
        every comfy-native quantised op (w4a8/int8/nvfp4/fp8) runs the EAGER
        fallback - weights dequantised to bf16 before each matmul. Measured
        A/B, one flag, same seed: 36.7 -> 17.6 s/it (-52%), and ~1.6x the
        activation pool. GGUF bypasses comfy-kitchen entirely and is unaffected.
        Say it once at load so nobody pays 2x for a checkpoint they chose for
        speed."""
        n = str(model_name).lower()
        if n.endswith(".gguf"):
            return
        try:
            import comfy_kitchen as ck
            b = ck.list_backends()
        except Exception:
            return
        def live(name):
            info = b.get(name) if isinstance(b, dict) else None
            return bool(info) and info.get("available") and not info.get("disabled")
        if live("cuda") or live("triton"):
            return
        print("[H3ModelLoader] NOTE: only comfy-kitchen's 'eager' backend is "
              "live on this box (cuda needs torch cu130+; triton needs "
              "--enable-triton-backend). Comfy-native quantised checkpoints "
              "(w4a8/int8/nvfp4/fp8) dequantise to bf16 every step here - "
              "measured ~2x slower and ~1.6x the activation pool. Either add "
              "--enable-triton-backend to your launch line, or use the GGUF "
              "build of this model, which is unaffected.", flush=True)

    def load(self, model_name, activation_reserve_gb=0.0):
        model_name = self._resolve_name(model_name)
        self._warn_quant_backend(model_name)
        out = self._load_inner(model_name)
        patcher = out[0]
        # stash the checkpoint name so samplers can check task compatibility
        # (fl2va = first/last-frame hand-off, ref2va = reference rows). The
        # wrong pairing does not error - it silently underperforms.
        try:
            patcher.model.h3_checkpoint_name = str(model_name)
        except Exception:
            pass
        if activation_reserve_gb and activation_reserve_gb > 0:
            _cap = int(activation_reserve_gb * (1024 ** 3))
            # Must live on the inner BaseModel, not the ModelPatcher: LoRA
            # stacks and guiders clone() the patcher before sampling and an
            # instance attribute does not survive the clone, silently
            # restoring comfy's estimate. Clones share this BaseModel.
            patcher.model.memory_required = lambda *a, _c=_cap, **k: _c
            patcher.memory_required = lambda *a, _c=_cap, **k: _c
            print(f"[H3ModelLoader] activation reserve PINNED at "
                  f"{activation_reserve_gb:.1f} GB (manual - correct for one "
                  f"resolution only; 0 = auto adapts to any)", flush=True)
        else:
            _install_auto_reserve(patcher, model_name)
        return out

    def _load_inner(self, model_name):
        import folder_paths
        # With VALIDATE_INPUTS accepting any name, the not-found case lands
        # here instead of at queue time - so say it clearly, with the folder
        # ComfyUI actually searched, before either loader path gets a chance
        # to fail obscurely.
        try:
            _found = folder_paths.get_full_path("diffusion_models", model_name)
        except Exception:
            _found = None
        if not _found:
            import os as _os
            _roots = [d for d in folder_paths.get_folder_paths("diffusion_models")]
            _hit = None
            for _d in _roots:
                if _os.path.isfile(_os.path.join(_d, model_name)):
                    _hit = _os.path.join(_d, model_name)
                    break
            if not _hit:
                _msg = (
                    "[H3ModelLoader] model file not found: %r. Searched: %s. "
                    "Pick your model file in this node's dropdown (a workflow "
                    "saved on another machine remembers a name your models "
                    "folder does not have)." % (model_name, ", ".join(_roots)))
                raise _h3_fail(_msg, RuntimeError, "H3 model missing",
                               tag="H3ModelLoader")
        if model_name.lower().endswith(".gguf"):
            # resolve the live UnetLoaderGGUF from the global registry -
            # custom node packages load under mangled module names, so the
            # registry is the only stable handle.
            import nodes as core_nodes
            cls = core_nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
            if cls is None:
                _msg = "ComfyUI-GGUF not loaded - install/enable it and restart."
                raise _h3_fail(_msg, RuntimeError, "H3 GGUF missing",
                               tag="H3ModelLoader")
            # ComfyUI-GGUF rejects unknown architectures before reading any
            # tensor, and upstream does not know minimax_h3. Import-time
            # patching covers the packaged install; re-assert here in case
            # ComfyUI-GGUF loaded after us. The relative import only exists
            # in the packaged install - LOOSE-FILE installs (this file
            # dropped straight into custom_nodes/) have no parent package,
            # so fall back to doing the patch inline.
            try:
                from .h3_gguf_arch import ensure_minimax_arch
                ensure_minimax_arch()
            except ImportError:
                import sys as _sys
                for _m in list(_sys.modules.values()):
                    try:
                        if (_m is not None
                                and isinstance(getattr(_m, "IMG_ARCH_LIST",
                                                       None), set)
                                and hasattr(_m, "TXT_ARCH_LIST")):
                            if "minimax_h3" not in _m.IMG_ARCH_LIST:
                                _m.IMG_ARCH_LIST.add("minimax_h3")
                                print("[H3ModelLoader] taught ComfyUI-GGUF "
                                      "the 'minimax_h3' architecture (in "
                                      "memory, loose-file fallback)",
                                      flush=True)
                            break
                    except Exception:
                        continue
            return cls().load_unet(model_name)
        import comfy.sd
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (comfy.sd.load_diffusion_model(path),)
