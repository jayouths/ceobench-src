"""Analytics, monitoring, and operations tools."""

from typing import Dict, Optional
from . import _client


def get_social_posts(days: int = 7, limit: int = 50) -> Dict:
    """Get recent social media posts from customers.

    Args:
        days: Number of days to look back (default 7).
        limit: Maximum number of posts to return (default 50).

    Returns:
        Dict with social media post data.
    """
    return _client.call('get_social_posts', {'days': days, 'limit': limit})


def set_targeted_ops_spend(
    *,
    by_group: Optional[Dict[str, float]] = None,
    by_plan: Optional[Dict[str, float]] = None,
    by_group_plan: Optional[Dict[str, Dict[str, float]]] = None,
    by_customer: Optional[Dict[str, float]] = None,
) -> Dict:
    """Set targeted operations spend for one or more scopes.

    Args:
        by_group: Per-group daily spend, ``{group_id: amount}``.
        by_plan: Per-plan daily spend, ``{plan: amount}``.
        by_group_plan: Per-group-plan daily spend,
            ``{group_id: {plan: amount}}``.
        by_customer: Per-customer daily spend,
            ``{customer_id_as_string: amount}``.

    Returns:
        Dict with update confirmation.
    """
    args = {}
    if by_group is not None:
        args['by_group'] = by_group
    if by_plan is not None:
        args['by_plan'] = by_plan
    if by_group_plan is not None:
        args['by_group_plan'] = by_group_plan
    if by_customer is not None:
        args['by_customer'] = by_customer
    return _client.call('set_targeted_ops_spend', args)


def set_targeted_dev_spend(targeted_spend: Dict[str, float]) -> Dict:
    """Set per-group targeted development spend.

    Per-group dev spend ACCUMULATES a quality bonus daily. Persists after spending stops.

    Args:
        targeted_spend: {group_id: daily_amount} dict.

    Returns:
        Dict with update confirmation.
    """
    return _client.call('set_targeted_dev_spend', {'targeted_spend': targeted_spend})
