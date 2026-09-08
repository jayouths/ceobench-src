"""R&D research project tools."""

from typing import Dict
from . import _client


def start_research_project(tier: int) -> Dict:
    """Start an R&D research project.

    Tiers are independent and repeatable. Use list_research_projects() for
    the current costs and duration/quality distributions. Duration and quality
    boost are randomly sampled when a project starts.

    Args:
        tier: Research tier returned by list_research_projects().

    Returns:
        Dict with project start confirmation.
    """
    return _client.call('start_research_project', {'tier': tier})


def list_research_projects() -> Dict:
    """List all R&D research tiers and their status.

    Returns:
        Dict with all research project details.
    """
    return _client.call('list_research_projects')
