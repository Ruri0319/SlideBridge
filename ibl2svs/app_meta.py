from __future__ import annotations

import os
from datetime import datetime, timezone

APP_NAME = "SlideBridge"
APP_VERSION = os.getenv("SLIDEBRIDGE_VERSION", os.getenv("IBL2SVS_VERSION", "0.4.5"))
BUILD_REF = os.getenv("SLIDEBRIDGE_BUILD_REF", os.getenv("IBL2SVS_BUILD_REF", "dev"))
BUILD_TIME = os.getenv(
    "SLIDEBRIDGE_BUILD_TIME",
    os.getenv(
        "IBL2SVS_BUILD_TIME",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ),
)


def app_title() -> str:
    return f"{APP_NAME} {APP_VERSION}"


def build_label() -> str:
    return f"{APP_VERSION} ({BUILD_REF[:7]})"


def runtime_banner() -> str:
    return f"{APP_NAME} {APP_VERSION} | build={BUILD_REF} | built={BUILD_TIME}"
