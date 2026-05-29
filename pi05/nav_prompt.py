"""Navigation intent categories for π0.5 driving.

VLM-based labeling (Gemini Flash) replaces trajectory-based heuristics.
These categories match what a GPS/navigation API provides at inference time.
"""

from __future__ import annotations

NAV_CATEGORIES = [
    "continue straight",
    "turn left",
    "turn right",
    "change lanes left",
    "change lanes right",
    "stop",
    "u-turn",
]
