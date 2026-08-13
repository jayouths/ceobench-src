"""Event logging for SaaS Bench simulation runs.

Provides detailed JSON logging of all simulation events including:
- Simulator events (shocks, outages, customer actions)
- Agent actions and their effects
- LLM API calls and costs (simulation-side)
- Daily state changes
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from .json_io import write_json_atomic


@dataclass
class EventLogEntry:
    """A single event log entry."""
    timestamp: str
    day: int
    event_type: str  # simulator, agent_action, llm_call, state_change, shock, outage, customer, etc.
    category: str    # More specific category within event_type
    details: Dict[str, Any]
    cost_usd: Optional[float] = None  # LLM cost if applicable

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class RunMetadata:
    """Metadata for a simulation run."""
    run_id: str
    seed: int
    scenario: str
    start_time: str
    config: Dict[str, Any]
    end_time: Optional[str] = None
    final_cash: Optional[float] = None
    days_run: Optional[int] = None
    total_llm_cost: Optional[float] = None
    outcome: Optional[str] = None  # 'completed', 'bankrupt', 'budget_exceeded'


class EventLogger:
    """JSONL event logger for simulation runs.

    Streams events to a JSONL file (one JSON object per line) immediately
    on each log call. No in-memory buffering — safe for long runs.

    Files:
        run_{id}.jsonl  — one event per line
        run_{id}_meta.json — metadata (written at start + end)
    """

    def __init__(
        self,
        run_id: str,
        output_dir: Path,
        seed: int,
        scenario: str,
        config: Dict[str, Any],
        starting_llm_cost_usd: float = 0.0,
        start_time: Optional[str] = None,
    ):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / f"run_{run_id}.jsonl"
        self.meta_file = self.output_dir / f"run_{run_id}_meta.json"

        self.metadata = RunMetadata(
            run_id=run_id,
            seed=seed,
            scenario=scenario,
            start_time=start_time or self._now(),
            config=config
        )

        if starting_llm_cost_usd < 0:
            raise ValueError("starting_llm_cost_usd must be non-negative")
        # 日期必须由事件发生处显式传入，避免并发和断点恢复后读到过期状态。
        self._accumulated_llm_cost_usd = float(starting_llm_cost_usd)
        self._event_count = 0

        # Open JSONL file for streaming writes
        self._file = open(self.log_file, 'a')

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _compact_result(result: Any, max_chars: int = 500) -> Any:
        """Keep structured small results; compact oversized values for JSONL safety."""
        try:
            serialized = json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(result)
        if len(serialized) <= max_chars:
            return result
        # 大结果只用于运行审计，日志中限长避免长期实验无界增长。
        return serialized[:max_chars] + "..."

    def _write_event(self, entry: EventLogEntry):
        """Write a single event to disk immediately."""
        self._file.write(json.dumps(entry.to_dict()) + "\n")
        self._event_count += 1
        # Flush every 10 events to balance performance and durability
        if self._event_count % 10 == 0:
            self._file.flush()

    # =========================================================================
    # Simulator Events
    # =========================================================================

    def log_shock(self, day: int, shock_type: str, details: Dict[str, Any]):
        """Log a shock event (demand_surge, etc.)."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="shock",
            details={
                "shock_type": shock_type,
                **details
            }
        ))

    def log_outage(self, day: int, downtime_minutes: int, overload: float):
        """Log a service outage event."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="outage",
            details={
                "downtime_minutes": downtime_minutes,
                "overload": overload
            }
        ))

    def log_customer_signup(self, day: int, customer_id: int, group_id: str, plan: str,
                           price: float, is_enterprise: bool, seat_count: Optional[int] = None):
        """Log a new customer subscription."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="customer_signup",
            details={
                "customer_id": customer_id,
                "group_id": group_id,
                "plan": plan,
                "price": price,
                "is_enterprise": is_enterprise,
                "seat_count": seat_count
            }
        ))

    def log_customer_churn(self, day: int, customer_id: int, group_id: str, plan: str,
                          reason: str, satisfaction: Optional[float] = None):
        """Log a customer cancellation."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="customer_churn",
            details={
                "customer_id": customer_id,
                "group_id": group_id,
                "plan": plan,
                "reason": reason,
                "satisfaction": satisfaction
            }
        ))

    def log_plan_change(self, day: int, customer_id: int, old_plan: str, new_plan: str,
                       old_price: float, new_price: float, direction: str):
        """Log a plan upgrade or downgrade."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="plan_change",
            details={
                "customer_id": customer_id,
                "old_plan": old_plan,
                "new_plan": new_plan,
                "old_price": old_price,
                "new_price": new_price,
                "direction": direction  # 'upgrade' or 'downgrade'
            }
        ))

    def log_social_post(self, day: int, customer_id: int, group_id: str, sentiment: str,
                       virality_score: float, reputation_impact: float):
        """Log a social media post and its impact."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="social_post",
            details={
                "customer_id": customer_id,
                "group_id": group_id,
                "sentiment": sentiment,
                "virality_score": virality_score,
                "reputation_impact": reputation_impact
            }
        ))

    def log_negotiation_event(self, day: int, thread_id: int, customer_id: int, event: str,
                             details: Dict[str, Any]):
        """Log an enterprise negotiation event."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="negotiation",
            details={
                "thread_id": thread_id,
                "customer_id": customer_id,
                "event": event,  # 'started', 'offer', 'counter_offer', 'accepted', 'rejected', 'timeout'
                **details
            }
        ))

    def log_deal_closed(self, day: int, customer_id: int, thread_id: int, thread_type: str,
                        agreed_price: float, seat_count: int):
        """Log an enterprise deal closure."""
        self.log_negotiation_event(
            day=day,
            thread_id=thread_id,
            customer_id=customer_id,
            event="deal_closed",
            details={
                "thread_type": thread_type,
                "agreed_price": agreed_price,
                "seat_count": seat_count,
            }
        )

    def log_issue(self, day: int, customer_id: int, event: str, days_open: Optional[int] = None):
        """Log a customer issue event."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="simulator",
            category="issue",
            details={
                "customer_id": customer_id,
                "event": event,  # 'opened', 'resolved', 'escalated'
                "days_open": days_open
            }
        ))

    # =========================================================================
    # Agent Actions
    # =========================================================================

    def log_agent_action(self, day: int, tool_name: str, arguments: Dict[str, Any],
                        result: Any, success: bool):
        """Log an agent tool call and its result."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="agent_action",
            category=tool_name,
            details={
                "arguments": arguments,
                "result": self._compact_result(result),
                "success": success
            }
        ))

    def log_agent_turn(self, day: int, turn: int, model: str, input_tokens: int,
                      output_tokens: int, tool_calls: List[str]):
        """Log an agent turn (one LLM call cycle)."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="agent_action",
            category="turn",
            details={
                "turn": turn,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tool_calls": tool_calls
            }
        ))

    # =========================================================================
    # LLM Calls (Simulation-side)
    # =========================================================================

    def log_llm_call(self, day: int, purpose: str, model: str, input_tokens: int,
                    output_tokens: int, cost_usd: float, details: Optional[Dict] = None):
        """Log a simulation-side LLM call (customer simulation, negotiations, etc.)."""
        self._accumulated_llm_cost_usd += cost_usd

        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="llm_call",
            category=purpose,  # 'social_post', 'negotiation_response', 'initial_message', etc.
            details={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                **(details or {})
            },
            cost_usd=cost_usd
        ))

    # =========================================================================
    # State Changes
    # =========================================================================

    def log_daily_state(self, day: int, cash: float, mrr: float, subscribers: int,
                       usage: int, overload: float, outage: bool,
                       group_reputations: Dict[str, float],
                       group_awareness: Dict[str, float],
):
        """Log end-of-day state snapshot."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="state_change",
            category="daily_snapshot",
            details={
                "cash": cash,
                "mrr": mrr,
                "subscribers": subscribers,
                "usage": usage,
                "overload": overload,
                "outage": outage,
                "group_reputations": group_reputations,
                "group_awareness": group_awareness,
            }
        ))

    def log_config_change(self, day: int, field: str, old_value: Any, new_value: Any, source: str):
        """Log a configuration change."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="state_change",
            category="config_change",
            details={
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
                "source": source  # 'agent' or 'system'
            }
        ))

    # =========================================================================
    # Run Lifecycle
    # =========================================================================

    def log_run_start(self):
        """Log the start of the simulation run."""
        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=0,
            event_type="lifecycle",
            category="run_start",
            details={
                "run_id": self.run_id,
                "seed": self.metadata.seed,
                "scenario": self.metadata.scenario
            }
        ))

    def log_run_end(self, day: int, final_cash: float, days_run: int, outcome: str):
        """Log the end of the simulation run."""
        self.metadata.end_time = self._now()
        self.metadata.final_cash = final_cash
        self.metadata.days_run = days_run
        self.metadata.total_llm_cost = self._accumulated_llm_cost_usd
        self.metadata.outcome = outcome

        self._write_event(EventLogEntry(
            timestamp=self._now(),
            day=day,
            event_type="lifecycle",
            category="run_end",
            details={
                "final_cash": final_cash,
                "days_run": days_run,
                "outcome": outcome,
                "total_llm_cost": self._accumulated_llm_cost_usd
            }
        ))

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self):
        """Flush JSONL and write metadata file."""
        self._file.flush()
        write_json_atomic(self.meta_file, asdict(self.metadata))

    def save_incremental(self):
        """Flush pending writes to disk."""
        self._file.flush()

    def close(self):
        """Close the JSONL file handle."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __del__(self):
        self.close()

    @staticmethod
    def generate_run_id() -> str:
        return str(uuid.uuid4())[:8]

    @classmethod
    def load(cls, log_file: Path) -> 'EventLogger':
        """Load an existing JSONL log file."""
        meta_file = log_file.parent / log_file.name.replace('.jsonl', '_meta.json')
        # Try loading metadata from meta file or fall back to old JSON format
        if meta_file.exists():
            with open(meta_file) as f:
                metadata = json.load(f)
        else:
            # Legacy: try loading from old .json format
            old_json = log_file.with_suffix('.json')
            if old_json.exists():
                with open(old_json) as f:
                    data = json.load(f)
                metadata = data['metadata']
            else:
                raise FileNotFoundError(f"No metadata file found for {log_file}")

        logger = cls(
            run_id=metadata['run_id'],
            output_dir=log_file.parent,
            seed=metadata['seed'],
            scenario=metadata['scenario'],
            config=metadata['config'],
            starting_llm_cost_usd=metadata.get('total_llm_cost') or 0.0,
        )
        logger.metadata = RunMetadata(**metadata)
        return logger
