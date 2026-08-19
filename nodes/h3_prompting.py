# -*- coding: utf-8 -*-
"""Prompt/reference text helpers for H3 Multishot Advance."""

import re


def _parse_ref_groups(spec, n_image):
    """"3,3" -> [1,1,1,2,2,2]: which subject each reference picture belongs to.

    Empty/invalid returns None, meaning "every picture is <Subject 1>" - the
    behaviour every release before 2.2.4 had. Counts that do not add up to
    n_image are corrected rather than rejected: a trailing shortfall joins the
    last subject, an overshoot is truncated. A user who mutes a reference image
    should not get a hard error out of a text field.
    """
    if not spec or not str(spec).strip() or n_image <= 0:
        return None
    try:
        counts = [int(x) for x in re.split(r"[,\s]+", str(spec).strip()) if x]
    except ValueError:
        print("[H3] reference_subjects %r is not a comma list of counts - "
              "treating every reference as one subject." % spec, flush=True)
        return None
    counts = [c for c in counts if c > 0]
    if len(counts) < 2:
        return None
    out = []
    for si, c in enumerate(counts, 1):
        out.extend([si] * c)
    if len(out) < n_image:
        out.extend([out[-1]] * (n_image - len(out)))
    return out[:n_image]


def _subject_defs(n_image, n_audio, n_video, speaker="the person",
                  image_subjects=None, speaking=True):
    """Official H3 ref2va subject_definitions + retention_analysis block.

    The tokenizer emits reference items as bare "<Picture k>: ",
    "<Audio j>: " and "<Video k>: " labels BEFORE the prompt text
    (comfy/text_encoders/minimax.py). Without a subject_definitions section
    the model gets labelled references and is never told what they are or
    what to keep - which is why identity, room, colour and especially VOICE
    drift between chained shots. Syntax follows the MiniMax-H3 model card's
    Ref2VA case; retention keywords are fully_preserved / partially_copy /
    reference.

    image_subjects (2.2.4) is an optional list, one entry per reference
    picture, giving the subject number that picture belongs to. Until 2.2.4
    every picture was declared a photograph of <Subject 1> unconditionally, so
    references for two different people told the model all of them showed the
    same individual and it rendered the average - reported on Civitai as
    "the video doesn't contain anything that resembles the reference people".
    None keeps the old single-subject behaviour exactly.
    """
    import os as _os
    if _os.environ.get("H3_NO_SUBJECT_DEFS"):   # A/B switch for testing
        return ""
    if not (n_image or n_audio or n_video):
        return ""
    subs = list(image_subjects or [])
    n_sub = max(subs) if subs else 1
    # KursatAs 2026-08-19 04:45: silent shots must not receive a hidden
    # "Subject is speaking" instruction from auto subject_definitions.
    if speaking:
        subject_1 = "<Subject 1> is %s speaking in this scene." % speaker
    else:
        subject_1 = ("<Subject 1> is %s appearing in this scene without "
                     "speaking." % speaker)
    if n_sub <= 1:
        d = ["subject_definitions:", subject_1]
    else:
        # Only <Subject 1> is described as speaking; H3's voice conditioning is
        # single-speaker and naming several speakers competes for the audio lane.
        d = ["subject_definitions:", subject_1]
        for s in range(2, n_sub + 1):
            d.append("<Subject %d> is a different individual who also appears "
                     "in this scene." % s)
    # Do NOT enumerate accessories here. This block is unconditional text on a
    # BasicGuider path - cfg 1.0, no negative branch - so anything named is
    # ADDED and can never be subtracted by the user's prompt. Naming "glasses"
    # made every ref2va render force thick frames onto the subject no matter
    # what the prompt asked for, and "remove the glasses" only put the word in
    # the conditioning a second time (reported on Civitai 2026-08-13).
    # Face, skin, hair and wardrobe are identity; eyewear, hats and jewellery
    # are wardrobe choices that belong to the prompt.
    r = ["retention_analysis:",
         "<Subject 1> (appears in [Shot 1]): fully_preserved - <Subject 1> "
         "retains the same face, skin and hair, and "
         "stays in the same room under the same lighting and colour "
         "temperature."]
    for s in range(2, n_sub + 1):
        r.append("<Subject %d> (appears in [Shot 1]): fully_preserved - "
                 "<Subject %d> retains their own distinct face, skin and hair "
                 "and is never blended with <Subject 1>." % (s, s))
    for k in range(1, n_image + 1):
        s = subs[k - 1] if k <= len(subs) else 1
        d.append("<Picture %d> is a reference photograph of <Subject %d>."
                 % (k, s))
    for k in range(1, n_video + 1):
        d.append("<Video %d> is a clip from an earlier moment of this same "
                 "continuous scene, showing <Subject 1> in the same place "
                 "under the same light." % k)
        r.append("<Video %d>: reference - the target video keeps the "
                 "framing, camera distance, room contents and colour "
                 "temperature of <Video %d>." % (k, k))
    for j in range(1, n_audio + 1):
        # KursatAs 2026-08-19 04:45: audio ref text now follows speech guard.
        # Speaking shots describe voice timbre; silent shots describe ambience
        # only, so the prompt text does not reintroduce dialogue.
        if speaking:
            d.append("<Audio %d> is a reference audio track containing "
                     "<Subject 1>'s speaking voice and scene ambience." % j)
            r.append("<Audio %d>: reference - the target audio references "
                     "the voice timbre in <Audio %d> so <Subject 1> speaks "
                     "with the same voice." % (j, j))
        else:
            d.append("<Audio %d> is reference environmental audio, not "
                     "dialogue or voiceover." % j)
            r.append("<Audio %d>: reference - the target audio may preserve "
                     "ambient sound texture, but <Subject 1> remains "
                     "non-speaking." % j)
    return "\n".join(d) + "\n" + "\n".join(r)
