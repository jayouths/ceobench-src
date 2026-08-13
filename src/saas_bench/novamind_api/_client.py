"""HTTP client for communicating with the NovaMind API server."""

import sys
from typing import Any, Dict, Optional

from ._transport import HTTPTransportError, request_json


class NovaMindAPIError(Exception):
    """Raised when an API call fails."""
    pass


class _Vars:
    """Namespace for simulator variables (e.g., current_day)."""

    @property
    def current_day(self) -> int:
        """Get the current simulation day."""
        data = get_vars()
        return data.get('current_day', 0)


def _request(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call the API through Unix Socket when available, otherwise localhost TCP."""
    try:
        return request_json(method, path, body)
    except HTTPTransportError as exc:
        if isinstance(exc.payload, dict):
            raise NovaMindAPIError(exc.payload.get('error', f'HTTP {exc.status}'))
        text = exc.body.decode('utf-8', errors='replace')[:500]
        raise NovaMindAPIError(f"HTTP {exc.status}: {text}")
    except (ConnectionError, OSError, ValueError) as exc:
        raise NovaMindAPIError(f"Failed to connect to API server: {exc}")


def call(tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call a tool on the API server and return the result.

    Args:
        tool_name: Name of the tool to call
        args: Tool arguments as a dict

    Returns:
        Dict with the structured result data

    Raises:
        NovaMindAPIError: On failure (also prints to stderr)
    """
    result = _request('POST', '/call', {"tool": tool_name, "args": args or {}})

    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        raise NovaMindAPIError(error_msg)

    return result.get('data', {})


def next_week(predictions: Dict[str, Any] = None, rationale: str = None) -> Dict[str, Any]:
    """Advance the simulator by one week (7 days).

    Args:
        predictions: Required dict with one entry per horizon. Required keys:
            ``cash_1wk``, ``cash_4wk``, ``cash_12wk``, ``cash_26wk`` — the
            agent's cash forecast at +7d, +28d, +84d, +182d (~6 months).
            Each value must be a dict with three numeric fields:
            ``point`` (point estimate), ``lower`` (95% CI lower bound),
            ``upper`` (95% CI upper bound). Constraint per horizon:
            ``lower <= point <= upper``. Server returns 400 on any
            missing key, missing field, non-numeric value, or violated constraint.
        rationale: Required non-empty string capturing your strategic reasoning
            for this week's actions. Replaces the old standalone log_rationale
            tool. Server returns 400 if missing or empty.

    Returns:
        Dict with 'day' and 'dashboard' keys
    """
    if predictions is None:
        raise NovaMindAPIError(
            "next_week() requires 'predictions' dict with keys "
            "cash_1wk, cash_4wk, cash_12wk, cash_26wk; each value an object "
            "{point, lower, upper} (95% CI bounds in dollars)."
        )
    if not isinstance(rationale, str) or not rationale.strip():
        raise NovaMindAPIError(
            "next_week() requires a non-empty 'rationale' string capturing "
            "your strategic reasoning for this week's actions."
        )

    def _entry(p):
        return {
            "point": float(p["point"]),
            "lower": float(p["lower"]),
            "upper": float(p["upper"]),
        }

    body = {
        "rationale": rationale,
        "predictions": {
            "cash_1wk":  _entry(predictions["cash_1wk"]),
            "cash_4wk":  _entry(predictions["cash_4wk"]),
            "cash_12wk": _entry(predictions["cash_12wk"]),
            "cash_26wk": _entry(predictions["cash_26wk"]),
        },
    }
    result = _request('POST', '/next-week', body)

    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        raise NovaMindAPIError(error_msg)

    return result




def query(sql: str) -> Dict[str, Any]:
    """Execute a read-only SQL query against the simulator database.

    Hidden columns and internal tables are automatically filtered.
    Write queries are blocked — use the novamind_api functions instead.

    Args:
        sql: SQL SELECT query string.

    Returns:
        Dict with 'columns' (list of column names), 'rows' (list of dicts),
        and 'row_count' (int).

    Raises:
        NovaMindAPIError: On failure (blocked query, syntax error, etc.)
    """
    result = _request('POST', '/query', {"sql": sql})

    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        raise NovaMindAPIError(error_msg)

    # Print enum value hints to stderr so agent sees them
    hint = result.get('hint')
    if hint:
        print(f"\n⚠️  {hint}", file=sys.stderr)

    # Print truncation warning to stderr so agent knows to narrow the query
    warning = result.get('warning')
    if warning:
        print(f"\n⚠️  {warning}", file=sys.stderr)

    return result


def get_vars() -> Dict[str, Any]:
    """Get simulator variables."""
    return _request('GET', '/vars')


def _post(path: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generic POST to the API server."""
    return _request('POST', path, body or {})


def _get(path: str) -> Dict[str, Any]:
    """Generic GET from the API server."""
    return _request('GET', path)


def _delete(path: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generic DELETE to the API server."""
    return _request('DELETE', path, body or {})


# Singleton vars instance
vars = _Vars()
