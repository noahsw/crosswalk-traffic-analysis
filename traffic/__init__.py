"""Crosswalk traffic study: detection, tracking, counting and reporting."""

import logging
import os

# Ultralytics checks that the numpy it was tested with is installed when it loads a CoreML
# model, and tries to pip install it otherwise. uv manages the packages here, and the newer
# numpy works, so turn that off (before Ultralytics is imported) and drop its warning.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


class _NoNumpyCheck(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "requirement ['numpy" not in record.getMessage()


logging.getLogger("ultralytics").addFilter(_NoNumpyCheck())
