"""Minimal, self-contained replacement for ``ultralytics.nn.tasks.parse_model``.

Ultralytics resolves a YAML entry through ``globals()[name]`` and only knows how
to bookkeep channels for the module classes hard-coded in its own
``parse_model``.  Our three modules therefore cannot be added without touching
that function.  Instead of editing site-packages (fragile, hard to review), we
ship this copy and install it at import time from ``models/register.py``.

The logic is a transcription of ``ultralytics==8.2.103`` with one addition: a
registry lookup that lets ``GAMS``, ``DPS-Conv`` and ``GPDA`` reuse the standard
``(c1, c2, *args)`` channel rule.  Everything else - including the special
cases for ``Concat``, ``Detect``, ``RTDETRDecoder``, ... - is unchanged.

Only ``parse_model`` is replaced. The loss and assigner are not modified.
"""

import ast
import contextlib

import torch
import torch.nn as nn

import ultralytics.nn.tasks as _tasks
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.ops import make_divisible

# name -> class, filled by ``install()``
_CUSTOM = {}

# Classes that Ultralytics builds as ``m(c1, c2, *args)`` and whose output
# channel count is ``args[0]`` after width scaling.
_C1C2_NAMES = (
    "Classify", "Conv", "ConvTranspose", "GhostConv", "Bottleneck", "GhostBottleneck",
    "SPP", "SPPF", "DWConv", "Focus", "BottleneckCSP", "C1", "C2", "C2f",
    "RepNCSPELAN4", "ELAN1", "ADown", "AConv", "SPPELAN", "C2fAttn", "C3", "C3TR",
    "C3Ghost", "C3x", "RepC3", "PSA", "SCDown", "C2fCIB",
)

# Classes that take the repeat count in ``args[2]``.
_REPEAT_NAMES = ("BottleneckCSP", "C1", "C2", "C2f", "C2fAttn", "C3", "C3TR", "C3Ghost", "C3x", "RepC3", "C2fCIB")

# Detection heads that receive the input-channel list.
_HEAD_NAMES = ("Detect", "WorldDetect", "Segment", "Pose", "OBB", "ImagePoolingAttn", "v10Detect")


def install(custom_modules):
    """Register ``custom_modules`` and swap ``parse_model`` into Ultralytics."""
    _CUSTOM.clear()
    _CUSTOM.update(custom_modules)
    _tasks.parse_model = parse_model
    return parse_model


def _get(name):
    """Return an attribute of ``ultralytics.nn.tasks`` or ``None``."""
    return getattr(_tasks, name, None)


def _resolve(name):
    """Resolve a YAML module name to a class (custom registry first)."""
    if name in _CUSTOM:
        return _CUSTOM[name]
    if "nn." in name:
        return getattr(torch.nn, name[3:])
    return getattr(_tasks, name)


def _module_sets():
    """Build the channel-handling sets for the installed Ultralytics version."""
    c1c2 = {_get(n) for n in _C1C2_NAMES if _get(n) is not None}
    c1c2 |= {nn.ConvTranspose2d}
    dwc_transpose = _get("DWConvTranspose2d")
    if dwc_transpose is not None:
        c1c2.add(dwc_transpose)
    c1c2 |= set(_CUSTOM.values())

    repeat = {_get(n) for n in _REPEAT_NAMES if _get(n) is not None}
    heads = {_get(n) for n in _HEAD_NAMES if _get(n) is not None}
    return c1c2, repeat, heads


def parse_model(d, ch, verbose=True):
    """Parse a model YAML dict into a ``nn.Sequential`` (Ultralytics compatible)."""
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))

    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        conv_cls = _get("Conv")
        conv_cls.default_act = eval(act)  # noqa: S307 - same as Ultralytics
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    c1c2, repeat, heads = _module_sets()
    batch_norm = nn.BatchNorm2d
    concat_cls = _get("Concat")
    aifi = _get("AIFI")
    hg_stem, hg_block = _get("HGStem"), _get("HGBlock")
    resnet_layer = _get("ResNetLayer")
    rtdetr_decoder = _get("RTDETRDecoder")
    cblearn, cbfuse = _get("CBLinear"), _get("CBFuse")
    segment_cls = _get("Segment")

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = _resolve(m)  # noqa: PLW2901
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m in c1c2:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat:
                args.insert(2, n)
                n = 1
        elif aifi is not None and m is aifi:
            args = [ch[f], *args]
        elif (hg_stem is not None and m is hg_stem) or (hg_block is not None and m is hg_block):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if hg_block is not None and m is hg_block:
                args.insert(4, n)
                n = 1
        elif resnet_layer is not None and m is resnet_layer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is batch_norm:
            args = [ch[f]]
        elif concat_cls is not None and m is concat_cls:
            c2 = sum(ch[x] for x in f)
        elif m in heads:
            args.append([ch[x] for x in f])
            if segment_cls is not None and m is segment_cls:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif rtdetr_decoder is not None and m is rtdetr_decoder:
            args.insert(1, [ch[x] for x in f])
        elif cblearn is not None and m is cblearn:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif cbfuse is not None and m is cbfuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)
