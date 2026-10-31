"""Register the AW-GeoNet modules with Ultralytics.

Importing ``models`` runs :func:`install`, which

1. replaces ``ultralytics.nn.tasks.parse_model`` with the patched copy in
   ``models/parse_model.py`` (so the YAML names ``MSBlock``, ``GSConv`` and
   ``DeformableC2f`` are understood), and
2. registers the module classes under the import paths the released checkpoint
   was pickled with (``ultralytics.nn.modules.ms_block``, ...), so that
   ``YOLO("awgeonet_s_best.pt")`` unpickles without editing site-packages.

Both steps are idempotent.
"""

import sys

import ultralytics.nn.modules as ultra_modules

from .parse_model import install as _install_parse_model

from . import modules as _module_impl
from .modules import (
    ChannelShuffle,
    DeformableAttention,
    DeformableC2f,
    DeformableConv2d,
    GSConv,
    GQL,
    MSBlock,
    SIBM,
    VoV_GSCSP,
    _MSBlockUnit,
)

#: Names usable in a model YAML.
CUSTOM_MODULES = {
    "MSBlock": MSBlock,
    "_MSBlockUnit": _MSBlockUnit,
    "SIBM": SIBM,
    "GQL": GQL,
    "GSConv": GSConv,
    "ChannelShuffle": ChannelShuffle,
    "VoV_GSCSP": VoV_GSCSP,
    "DeformableC2f": DeformableC2f,
    "DeformableAttention": DeformableAttention,
    "DeformableConv2d": DeformableConv2d,
}

#: Filenames the released checkpoints were pickled with.
_CHECKPOINT_ALIASES = ("ms_block", "gsconv_modules", "Deformable_Attention")

#: Target Ultralytics version for this model implementation.
EXPECTED_ULTRALYTICS = "8.2.103"

_APPLIED = False


def install(force=False):
    """Patch Ultralytics in-process.  Safe to call multiple times."""
    global _APPLIED
    if _APPLIED and not force:
        return

    _install_parse_model(CUSTOM_MODULES)

    for alias in _CHECKPOINT_ALIASES:
        # Deliberately overwrite (not setdefault): an internal development fork
        # ships a variant of these modules with ablation switches, and the
        # released checkpoints were pickled with the plain classes below.
        sys.modules[f"ultralytics.nn.modules.{alias}"] = _module_impl
        setattr(ultra_modules, alias, _module_impl)

    _check_version()
    _APPLIED = True


def _check_version():
    """Warn when the installed Ultralytics differs from the released one."""
    import warnings

    try:
        from ultralytics import __version__ as found
    except Exception:  # noqa: BLE001
        return
    if found != EXPECTED_ULTRALYTICS:
        warnings.warn(
            f"This implementation targets ultralytics=={EXPECTED_ULTRALYTICS}, found {found}. "
            f"Use the specified version for this implementation.",
            RuntimeWarning,
            stacklevel=3,
        )


def installed():
    """Return True once :func:`install` has run in this process."""
    return _APPLIED
