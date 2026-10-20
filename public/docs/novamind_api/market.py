"""Market discovery and research tools."""

from typing import Dict
from . import _client


def research_market() -> Dict:
    """Attempt to discover a new customer segment.

    Costs $25K per attempt, 30% chance of discovering a random hidden group.

    Returns:
        Dict with discovery result.
    """
    return _client.call('research_market')


def research_group(group_id: str, target_level: int | None = None) -> Dict:
    """Upgrade or refresh a discovered group's information.

    Higher info levels give more accurate data about the group.

    Args:
        group_id: The group identifier (e.g., 'D_S01').
        target_level: Optional level (2-5). Omit to upgrade Level 1 to 2,
            or refresh at the current level for Levels 2-5.

    Returns:
        Dict with research result.
    """
    return _client.call(
        'research_group',
        {'group_id': group_id, 'target_level': target_level},
    )


def get_market_overview() -> Dict:
    """Get overview of all known customer segments.

    Returns:
        Dict with market overview data.
    """
    return _client.call('get_market_overview')


def get_group_insights(group_id: str) -> Dict:
    """Get detailed insights for a specific customer group.

    Includes referral rates, reputation influence, segment characteristics.

    Args:
        group_id: The group identifier.

    Returns:
        Dict with group insight data.
    """
    return _client.call('get_group_insights', {'group_id': group_id})
