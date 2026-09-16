"""
Gateway ID normalisation — shared utility for consistent ID formatting.

All gateway IDs are normalised to 12-character uppercase hex (no colons).
Both colon-separated (AA:BB:CC:DD:EE:FF) and bare hex (AABBCCDDEEFF) formats
are supported as input.
"""

from __future__ import annotations

import re

_COLON_PATTERN = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def normalise_gateway_id(gw_id: str) -> str:
    """Convert any gateway ID format to 12-char uppercase hex (no colons).

    Examples:
        >>> normalise_gateway_id("AA:BB:CC:DD:EE:FF")
        'AABBCCDDEEFF'
        >>> normalise_gateway_id("aabbccddeeff")
        'AABBCCDDEEFF'
    """
    text = str(gw_id).strip()
    if _COLON_PATTERN.match(text):
        return text.replace(":", "").upper()
    # Already bare hex or close to it
    return re.sub(r"[^0-9A-Fa-f]", "", text).upper()


def format_colon_separated(gw_id: str) -> str:
    """Format a normalised gateway ID as colon-separated hex.

    Examples:
        >>> format_colon_separated("AABBCCDDEEFF")
        'AA:BB:CC:DD:EE:FF'
    """
    bare = normalise_gateway_id(gw_id)
    return ":".join(bare[i:i+2] for i in range(0, len(bare), 2))
