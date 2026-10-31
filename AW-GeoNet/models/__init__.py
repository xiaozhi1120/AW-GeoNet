"""AW-GeoNet model package.

Importing this package applies the Ultralytics patches, so a minimal usage is::

    import models  # noqa: F401
    from ultralytics import YOLO
    model = YOLO("models/awgeonet_s.yaml")
"""

from .register import CUSTOM_MODULES, EXPECTED_ULTRALYTICS, install, installed

install()

__all__ = ["CUSTOM_MODULES", "EXPECTED_ULTRALYTICS", "install", "installed"]
