# -*- coding: utf-8 -*-
"""Foreground-speech detection and prompt guarding for H3 multishot.

The goal is not language understanding. It is a conservative sampler-level
switch: only feed voice/audio references into a shot when the shot text clearly
asks the main subject to speak. Otherwise H3 can reuse a previous voice anchor
and invent dialogue in a random language.
"""

import re


# KursatAs 2026-08-19 04:45: sampler-level speech guard cues. Silent cues
# intentionally win over speech cues so "no dialogue" cannot be defeated by
# nearby words like "voice", "talking", or "lip movement" in a guard sentence.
_SILENT_PATTERNS = [
    r"\[(?:silent|no[_\s-]?speech|no[_\s-]?dialogue|no[_\s-]?voice)\]",
    r"\bno\s+(?:foreground\s+)?(?:dialogue|speech|spoken\s+words|voiceover|voice-over|talking)\b",
    r"\bwithout\s+(?:speaking|talking|dialogue|speech)\b",
    r"\b(?:does\s+not|do\s+not|don't|doesn't)\s+(?:speak|talk)\b",
    r"\b(?:lips|mouth)\s+(?:closed|remain\s+closed|stays\s+closed)\b",
    r"\bno\s+audible\s+(?:speech|dialogue|spoken\s+words)\b",
    r"\bambient\s+sound\s+only\b",
    r"\bnatural\s+environmental\s+sound\s+only\b",
    r"\bkonu[sş]ma\s+yok\b",
    r"\bdiyalog\s+yok\b",
    r"\bsessiz\b",
    r"\bkonu[sş]maz\b",
    r"\bkonu[sş]madan\b",
    r"\ba[gğ]z[ıi]\s+kapal[ıi]\b",
]

_SPEECH_PATTERNS = [
    r"\[(?:speech|dialogue|spoken|voiceover|voice-over|narration)\]",
    r"\b(?:dialogue|spoken\s+dialogue|line|voiceover|voice-over|narration)\s*:",
    r"\b(?:says|speaks|talks|whispers|shouts|asks|replies|responds|answers|tells|murmurs|utters|narrates)\b",
    r"\b(?:starts|begins|continues)\s+(?:speaking|talking|saying)\b",
    r"\b(?:speaking|talking)\s+(?:directly|calmly|softly|loudly|to\s+camera|to\s+the\s+camera)\b",
    r"\blip[-\s]?sync(?:ed|ing)?\b",
    r"\bkonu[sş](?:ur|uyor|maya\s+ba[sş]lar)\b",
    r"\bs[öo]yler\b",
    r"\bdiyor\b",
    r"\bder\s*:",
    r"\bf[ıi]s[ıi]ldar\b",
    r"\bba[gğ][ıi]r[ıi]r\b",
    r"\bcevap\s+verir\b",
    r"\banlat[ıi]r\b",
]

_QUOTE_PATTERN = re.compile(r"[\"“”‘’'][^\"“”‘’']{4,}[\"“”‘’']")

# KursatAs 2026-08-19 04:45: appended only on shots where no explicit
# foreground-speech cue exists. This keeps ambience available while stopping
# H3 from reusing an old voice anchor as invented dialogue.
_NO_SPEECH_GUARD = (
    "Speech guard: no foreground dialogue, no speech, no voiceover, no "
    "singing, and no lip-synced words from any visible subject. Keep only "
    "non-verbal action and ambient environmental sound; any background crowd "
    "murmur must remain unintelligible and non-linguistic."
)


def h3_detect_foreground_speech(prompt):
    """Return ``(has_speech, reason)`` for one shot prompt.

    Silent markers win over speech markers so prompts like "no speech, lips
    closed" cannot be flipped by nearby words such as "voice" or "talking" in
    an instruction sentence.
    """
    text = str(prompt or "")
    low = text.casefold()
    compact = re.sub(r"\s+", " ", low)

    for pattern in _SILENT_PATTERNS:
        if re.search(pattern, compact):
            return False, "silent cue"

    for pattern in _SPEECH_PATTERNS:
        if re.search(pattern, compact):
            return True, "speech cue"

    if _QUOTE_PATTERN.search(text):
        return True, "quoted line"

    return False, "no speech cue"


def h3_apply_no_speech_guard(prompt):
    """Append a hard no-speech instruction once."""
    text = str(prompt or "")
    if _NO_SPEECH_GUARD.casefold() in text.casefold():
        return text
    return text.rstrip() + "\n\n" + _NO_SPEECH_GUARD
