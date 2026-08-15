"""Analysis 模块的结构化数据契约与执行逻辑。"""

from .models import (
    Role,
    RoleAnalysis,
    RoleReport,
    StateAssessment,
    StatePortrait,
)
from .signal_models import AnalysisSignals
from .signal_catalog import SIGNAL_CATALOG, SIGNAL_CATALOG_VERSION
from .signals import SignalCollector, parse_public_week_snapshot

__all__ = [
    "Role",
    "RoleAnalysis",
    "RoleReport",
    "StateAssessment",
    "StatePortrait",
    "AnalysisSignals",
    "SignalCollector",
    "parse_public_week_snapshot",
    "SIGNAL_CATALOG",
    "SIGNAL_CATALOG_VERSION",
]
