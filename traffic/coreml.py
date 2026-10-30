"""YOLO as a CoreML model, so detection runs on a Mac's Neural Engine instead of the GPU.

It finds the same objects as the .pt model (to within rounding) in about half the time and
power. The model is exported once, from the .pt file of the same name, the first time it's used.
"""

from __future__ import annotations

import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np

ASPECT = 16 / 9   # frames are 1920x1080 (phone and camera alike); other shapes are padded to fit


def export(dst: Path, imgsz: int) -> None:
    """Export `dst` (an .mlpackage) from the .pt model beside it, `imgsz` wide."""
    from ultralytics import YOLO
    from .pipeline import load_model

    src = dst.with_suffix(".pt")
    load_model(str(src))   # downloads it if needed
    # A CoreML model's input size is fixed: the same size the .pt model resizes frames to.
    shape = shape_for(imgsz)
    _patch_coremltools()
    cwd = os.getcwd()
    os.chdir(src.parent)   # Ultralytics writes the export into the working directory
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 16-bit, because the Neural Engine doesn't run 32-bit models.
            out = Path(YOLO(src.name).export(format="coreml", imgsz=shape, half=True, nms=False)).resolve()
    finally:
        os.chdir(cwd)
    if out != dst.resolve():
        shutil.rmtree(dst, ignore_errors=True)
        out.rename(dst)


def shape_for(imgsz: int) -> tuple[int, int]:
    """(height, width) a 16:9 frame is resized to for a model `imgsz` wide."""
    return math.ceil(imgsz / ASPECT / 32) * 32, imgsz


def input_size(path: str) -> tuple[int, int]:
    """(height, width) of a CoreML model's image input."""
    import coremltools as ct
    image = ct.models.MLModel(path, skip_model_load=True).get_spec().description.input[0].type.imageType
    return image.height, image.width


def _patch_coremltools() -> None:
    """coremltools 9.0 casts one-element arrays with int(array), which numpy 2.5 no longer allows."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    original = ops._cast

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
            context.add(mb.const(val=dtype(x.val.item()), name=node.name), node.name)
        else:
            original(context, node, dtype, dtype_name)

    ops._cast = _cast
