"""
설정 파일 모듈

Sim2Real 공유 설정:
- PenWorkspaceConfig: 펜 작업 공간 설정 (위치/각도 유효 범위)
"""

from .pen_workspace import (
    PenWorkspaceConfig,
    DEFAULT_PEN_WORKSPACE,
    calculate_tilt_from_direction,
)

__all__ = [
    "PenWorkspaceConfig",
    "DEFAULT_PEN_WORKSPACE",
    "calculate_tilt_from_direction",
]
