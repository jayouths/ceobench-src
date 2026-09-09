"""Enterprise customer management tools."""

from typing import Dict, List
from . import _client


def send_enterprise_deal(deals: List) -> Dict:
    """Send enterprise deal offerings to one or more customers.

    Compact tuple format: deals=[[customer_id, [["plan", price_per_seat], ...]]]

    All contracts are month-to-month. Contract duration cannot be configured.

    If customer has an open thread, this replies; otherwise initiates renegotiation.

    Args:
        deals: List of [customer_id, [offerings]] pairs. Each customer ID may
            appear only once; duplicates reject the entire call.

    Returns:
        Dict with per-customer results.
    """
    return _client.call('send_enterprise_deal', {'deals': deals})


def reject_enterprise_deal(deals: List) -> Dict:
    """Reject enterprise negotiation threads.

    WARNING: Rejecting renegotiation/renewal/churn_prevention threads causes churn.

    Args:
        deals: List of customer deal rejection specs.

    Returns:
        Dict with per-customer results.
    """
    return _client.call('reject_enterprise_deal', {'deals': deals})
