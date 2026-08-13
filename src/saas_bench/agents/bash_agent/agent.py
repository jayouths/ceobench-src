"""Bash agent for SaaS Bench.

A Claude Code-style agent that uses bash and file tools to interact with the
NovaMind SaaS simulator via the novamind_api Python library and CLI.

Supports OpenAI-compatible APIs (OpenAI, xAI) and Anthropic APIs (direct, Bedrock).
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from ..base import BaseAgent
from ...environment import Action
from ...llm_provider import openai_chat_cached_tokens


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Any  # str or list (Anthropic content blocks)
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class BashAgent(BaseAgent):
    """Bash agent for SaaS Bench — Claude Code-style.

    Uses bash, read_file, write_file, edit_file, search_files, glob_files
    tools. Interacts with the simulator via novamind_api Python library
    and ./novamind-operation CLI.

    After calling `./novamind-operation next-week`, context is refreshed:
    the conversation is cleared and rebuilt with system prompt + MEMORY.md
    contents + the new dashboard.
    """

    CHECKPOINT_SNAPSHOT_FORMAT_VERSION = 2
    # 推进命令的完整参数只在权威 CLI 文档中维护，避免多处示例过期。
    NO_TOOL_FEEDBACK = (
        "You must call a tool to proceed. If you need context, use read_file, "
        "search_files, or bash. If you have nothing else to do this week, use bash "
        "to run `./novamind-operation next-week` with all required arguments. "
        "Read `docs/cli.md` for the current syntax."
    )

    def __init__(
        self,
        tool_descriptions: List[Dict[str, Any]],
        client,
        model: Optional[str] = None,
        api_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_invalid_responses_per_turn: Optional[int] = None,
        response_callback: Optional[callable] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout_seconds: float = 600.0,
        request_options: Optional[Dict[str, Any]] = None,
        tool_result_callback: Optional[callable] = None,
        workspace_path: Optional[Path] = None,
        total_days: int = 3650,
    ):
        super().__init__(tool_descriptions)
        if not model:
            raise ValueError("agent model must be explicitly configured")
        if not api_type:
            raise ValueError("agent api_type must be explicitly configured")
        if max_output_tokens is None:
            raise ValueError("agent max_output_tokens must be explicitly configured")
        if max_output_tokens <= 0:
            raise ValueError("agent max_output_tokens must be positive")
        if (
            not isinstance(max_invalid_responses_per_turn, int)
            or isinstance(max_invalid_responses_per_turn, bool)
            or max_invalid_responses_per_turn <= 0
        ):
            raise ValueError(
                "agent max_invalid_responses_per_turn must be explicitly configured as a positive integer"
            )
        self.client = client
        self.model = model
        self.api_type = api_type
        self.max_invalid_responses_per_turn = max_invalid_responses_per_turn
        self.response_callback = response_callback
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.request_options = dict(request_options or {})
        self.tool_result_callback = tool_result_callback
        self.workspace_path = workspace_path or Path('.')
        self.total_days = total_days

        self.use_anthropic = api_type == "anthropic_messages"

        # Build system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Agent state
        self.conversation: List[Message] = []
        self.current_day: int = 0
        self.turns_today: int = 0
        self._pending_tool_calls: List[Dict] = []
        self._last_observation: str = ""
        self.total_turns: int = 0
        self._consecutive_errors: int = 0

        # Token usage tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_cached_tokens: int = 0
        self.last_reasoning_tokens: int = 0
        self.last_serving_model: str = model

        # 每次 LLM 调用后的可读诊断快照；精确恢复只使用 Runner 提交的不可变 checkpoint。
        self._snapshot_path: Optional[Path] = None
        # checkpoint 已包含最后一个工具结果，恢复首轮不能再追加 Runner 新取的 Dashboard。
        self._skip_next_observation: bool = False

    def _default_system_prompt(self) -> str:
        """Build the default system prompt.

        Loads the bash_agent system_prompt.md and fills in
        {simulator_instructions} and {total_days}.

        ORACLE MODE: when env var ORACLE_MODE=1, prepend system_prompt_oracle.md
        as a preamble. The oracle preamble explicitly overrides the "hidden
        column / not accessible" caveats in the standard prompt, lists the
        hidden tables and customer_state columns, and points the agent at the
        read-only simulator source tree under /data/saas-bench/src.
        """
        base_dir = Path(__file__).parent

        # Load simulator instructions — strip the {tool_list} placeholder
        # (bash_agent has its own tool reference in system_prompt.md)
        simulator_file = base_dir.parent / "simulator_instructions.md"
        with open(simulator_file, 'r') as f:
            sim_text = f.read()

        sim_text = sim_text.replace('{tool_list}\n', '')
        sim_text = sim_text.replace('{tool_list}', '')

        template_file = base_dir / "system_prompt.md"
        with open(template_file, 'r') as f:
            template = f.read()

        prompt = template.replace('{simulator_instructions}', sim_text)

        # Replace {total_days} placeholder with actual value
        total_years = self.total_days / 365
        years_str = f"{total_years:.0f}" if total_years == int(total_years) else f"{total_years:.1f}"
        prompt = prompt.replace('{total_days}', str(self.total_days))
        prompt = prompt.replace('{total_years}', years_str)

        if os.environ.get("ORACLE_MODE") == "1":
            oracle_file = base_dir / "system_prompt_oracle.md"
            if oracle_file.exists():
                with open(oracle_file, 'r') as f:
                    oracle_preamble = f.read()
                prompt = oracle_preamble + "\n\n" + prompt
        return prompt

    def _get_system_prompt_with_memory(self) -> str:
        """Return system prompt with MEMORY.md contents appended.

        MEMORY.md is always injected into the system prompt so the agent
        has its persistent notes available without needing to read the file.
        """
        prompt = self.system_prompt
        memory_path = self.workspace_path / 'MEMORY.md'
        if memory_path.exists():
            # MEMORY.md 是跨周信息的唯一自动入口，读取失败不能静默降级。
            memory_content = memory_path.read_text().strip()
            if memory_content:
                max_memory_chars = 40_000
                if len(memory_content) > max_memory_chars:
                    memory_content = memory_content[:max_memory_chars] + (
                        "\n\n--- MEMORY.md TRUNCATED ---\n"
                        f"Showing first {max_memory_chars:,} of {len(memory_content):,} characters. "
                        "Use the read_file tool to see the full contents if needed."
                    )
                prompt += (
                    "\n\n## Your MEMORY.md (auto-loaded)\n\n"
                    "The following is the contents of your MEMORY.md file. "
                    "This is automatically loaded into your context at the start of every week.\n\n"
                    f"{memory_content}"
                )
        return prompt

    def reset(self):
        """Reset agent state for a new episode."""
        self.conversation = []
        self.current_day = 0
        self.turns_today = 0
        self._pending_tool_calls = []
        self._last_observation = ""

    def _reset_week_context(self) -> None:
        """Discard last week's conversation and rebuild the system context."""
        self.conversation = []
        self._pending_tool_calls = []

        if not self.use_anthropic:
            # OpenAI 协议把系统提示放入对话；Anthropic 在请求参数中单独传递。
            self.conversation.append(Message(
                role='system',
                content=self._get_system_prompt_with_memory(),
            ))

    def act(self, observation: str, reward: float, done: bool, info: Dict[str, Any]) -> Optional[Action]:
        """Choose an action based on the observation.

        The agent processes tool outputs and decides the next action.
        After week advancement is detected, context is reset.
        """
        if done:
            return None

        self._last_observation = observation

        # Day 0 是实验的合法首日；不能只用“日期变大”判断首次上下文初始化。
        current_day = info.get('day', 0)
        needs_initial_context = not self.conversation and not self._skip_next_observation
        if needs_initial_context or current_day > self.current_day:
            self._reset_week_context()
            self.current_day = current_day
            self.turns_today = 0

        # If we have pending tool call results to process, add them
        if self._skip_next_observation:
            # checkpoint 已把最后一次工具结果写入对话，恢复后的首轮不能重复追加 Dashboard。
            self._skip_next_observation = False
        elif self._pending_tool_calls:
            if self.use_anthropic:
                partial_results = self._pending_tool_calls[0].get('_partial_results', [])
                tool_results = [{
                    'type': 'tool_result',
                    'tool_use_id': self._pending_tool_calls[0]['id'],
                    'content': observation,
                }]
                tool_results.extend(partial_results)
                self.conversation.append(Message(
                    role='user',
                    content=tool_results,
                ))
            else:
                for tc in self._pending_tool_calls:
                    self.conversation.append(Message(
                        role='tool',
                        content=observation,
                        tool_call_id=tc['id'],
                        name=tc['name']
                    ))
            self._pending_tool_calls = []
        else:
            # Add observation as user message (e.g., initial dashboard)
            self.conversation.append(Message(
                role='user',
                content=observation
            ))

        # Call LLM
        action = self._call_llm()
        self.turns_today += 1

        # Persist conversation snapshot so a mid-day crash can be resumed
        # with the exact accumulated context. Best-effort: failure here must
        # not derail the run, just log.
        self._save_conversation_snapshot()

        return action

    def _serialize_content_item(self, item: Any) -> Any:
        """Convert a single content item to a JSON-safe form.

        Assistant messages from the OpenAI Responses API store raw pydantic
        output items (ResponseReasoningItem, ResponseFunctionToolCall, etc.)
        in `content`. These must round-trip via `.model_dump()` so reasoning
        summaries + function_calls survive the snapshot. Plain dicts and
        primitives pass through unchanged.
        """
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json", exclude_none=False, by_alias=True)
        if isinstance(item, dict):
            return item
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        # Anthropic content blocks may be objects without model_dump; fall back
        # to repr so we never crash. They won't be replay-correct, but the
        # snapshot is best-effort and Anthropic uses a different code path.
        return repr(item)

    def _serialize_message(self, m: "Message") -> Dict[str, Any]:
        content = m.content
        if isinstance(content, list):
            content = [self._serialize_content_item(x) for x in content]
        return {
            "role": m.role,
            "content": content,
            "tool_calls": m.tool_calls,
            "tool_call_id": m.tool_call_id,
            "name": m.name,
        }

    def _save_conversation_snapshot(self) -> None:
        """Atomically write self.conversation + minimal turn state to disk.

        Overwrites the same file each call (single snapshot, not append-only).
        On Modal/NFS, rename() is atomic — readers see either the old or new
        complete file, never a partial write.

        Pydantic Responses-API output items (reasoning summaries, function
        calls) are converted via `.model_dump()` so they survive the JSON
        round-trip — see _serialize_content_item.
        """
        if self._snapshot_path is None:
            return
        try:
            payload = {
                "conversation": [self._serialize_message(m) for m in self.conversation],
                "pending_tool_calls": list(self._pending_tool_calls),
                "current_day": self.current_day,
                "turns_today": self.turns_today,
                "total_turns": self.total_turns,
                "last_observation_preview": (self._last_observation or "")[:2000],
                "saved_at": time.time(),
            }
            tmp_path = self._snapshot_path.with_suffix(self._snapshot_path.suffix + ".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self._snapshot_path)
        except Exception as e:
            # Never let snapshot failure kill the run.
            print(f"[snapshot] WARN failed to save conversation snapshot: {e}")

    def save_checkpoint_snapshot(
        self,
        path: Path,
        *,
        resume_conversation: bool,
        pending_observation: Optional[str] = None,
    ) -> None:
        """Save the exact Agent context associated with one durable checkpoint."""
        if resume_conversation:
            conversation = list(self.conversation)
            pending = list(self._pending_tool_calls)
            if pending:
                if pending_observation is None:
                    raise ValueError(
                        "pending_observation is required when checkpointing a pending tool result"
                    )
                if self.use_anthropic:
                    partial_results = pending[0].get('_partial_results', [])
                    tool_results = [{
                        'type': 'tool_result',
                        'tool_use_id': pending[0]['id'],
                        'content': pending_observation,
                    }]
                    tool_results.extend(partial_results)
                    conversation.append(Message(role='user', content=tool_results))
                else:
                    for tool_call in pending:
                        conversation.append(Message(
                            role='tool',
                            content=pending_observation,
                            tool_call_id=tool_call['id'],
                            name=tool_call['name'],
                        ))
            current_day = self.current_day
            turns_today = self.turns_today
        else:
            # 周边界恢复应让下一轮从 Dashboard 构建全新上下文。
            conversation = []
            current_day = 0
            turns_today = 0

        payload = {
            "format_version": self.CHECKPOINT_SNAPSHOT_FORMAT_VERSION,
            "resume_conversation": resume_conversation,
            "tool_results_applied": True,
            "conversation": [self._serialize_message(message) for message in conversation],
            "pending_tool_calls": [],
            "current_day": current_day,
            "turns_today": turns_today,
            "total_turns": self.total_turns,
            "saved_at": time.time(),
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as file:
            json.dump(payload, file)
        os.replace(tmp_path, path)

    @classmethod
    def parse_checkpoint_snapshot(cls, path: Path) -> Dict[str, Any]:
        """Read one immutable checkpoint snapshot using its strict schema."""
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Checkpoint conversation snapshot is invalid JSON") from exc
        required = {
            "format_version", "resume_conversation", "tool_results_applied",
            "conversation", "pending_tool_calls", "current_day", "turns_today",
            "total_turns", "saved_at",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError(
                f"Checkpoint conversation fields must contain exactly: {sorted(required)}"
            )
        if payload["format_version"] != cls.CHECKPOINT_SNAPSHOT_FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint conversation format_version")
        if not isinstance(payload["resume_conversation"], bool):
            raise ValueError("Checkpoint conversation resume_conversation must be boolean")
        if payload["tool_results_applied"] is not True:
            raise ValueError("Checkpoint conversation must include applied tool results")
        if payload["pending_tool_calls"] != []:
            raise ValueError("Checkpoint conversation cannot contain pending tool calls")
        for field in ("current_day", "turns_today", "total_turns"):
            value = payload[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid checkpoint conversation {field}: {value!r}")
        if not isinstance(payload["saved_at"], (int, float)) or isinstance(
            payload["saved_at"], bool
        ):
            raise ValueError("Checkpoint conversation saved_at must be numeric")
        if not isinstance(payload["conversation"], list):
            raise ValueError("Checkpoint conversation messages must be a list")
        allowed_roles = {"system", "user", "assistant", "tool"}
        required_message_fields = {
            "role", "content", "tool_calls", "tool_call_id", "name"
        }
        for index, message in enumerate(payload["conversation"]):
            if not isinstance(message, dict) or set(message) != required_message_fields:
                raise ValueError(
                    f"Invalid checkpoint conversation message at index {index}"
                )
            if message["role"] not in allowed_roles:
                raise ValueError(
                    f"Invalid checkpoint conversation role at index {index}"
                )
        return payload

    def restore_checkpoint_snapshot(self, payload: Dict[str, Any]) -> None:
        """Apply a snapshot already validated before simulator restoration."""
        self.conversation = [
            Message(
                role=message["role"],
                content=message["content"],
                tool_calls=message["tool_calls"],
                tool_call_id=message["tool_call_id"],
                name=message["name"],
            )
            for message in payload["conversation"]
        ]
        self._pending_tool_calls = []
        self.current_day = payload["current_day"]
        self.turns_today = payload["turns_today"]
        self._skip_next_observation = payload["resume_conversation"]

    def _call_llm(self) -> Optional[Action]:
        """Call the LLM and parse the response into an action."""
        if self.api_type == "anthropic_messages":
            return self._call_anthropic()
        if self.api_type == "openai_responses":
            return self._call_openai_responses()
        if self.api_type == "openai_chat_completions":
            return self._call_openai()
        raise ValueError(f"Unsupported decision-agent api_type: {self.api_type!r}")

    def _build_openai_chat_kwargs(self, messages, tools) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'tools': tools,
            'tool_choice': 'auto',
            'max_completion_tokens': self.max_output_tokens,
        }
        if self.temperature is not None:
            params['temperature'] = self.temperature
        if self.top_p is not None:
            params['top_p'] = self.top_p
        if self.reasoning_effort is not None:
            params['reasoning_effort'] = self.reasoning_effort
        params.update(self.request_options)
        return params

    def _build_openai_responses_kwargs(self, input_items, tools) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            'model': self.model,
            'input': input_items,
            'tools': tools,
            'tool_choice': 'auto',
            'max_output_tokens': self.max_output_tokens,
            'instructions': self._get_system_prompt_with_memory(),
        }
        if self.temperature is not None:
            params['temperature'] = self.temperature
        if self.top_p is not None:
            params['top_p'] = self.top_p
        if self.reasoning_effort is not None:
            params['reasoning'] = {'effort': self.reasoning_effort, 'summary': 'auto'}
        params.update(self.request_options)
        return params

    def _call_openai(self) -> Optional[Action]:
        """Call OpenAI-compatible API and parse the response."""
        import signal
        import openai

        LLM_WALL_CLOCK_TIMEOUT = max(1, int(self.timeout_seconds))

        class LLMTimeoutError(Exception):
            pass

        def _llm_timeout_handler(signum, frame):
            raise LLMTimeoutError(f"LLM call exceeded {LLM_WALL_CLOCK_TIMEOUT}s wall-clock timeout")

        invalid_responses = 0
        while True:
            messages = []
            for msg in self.conversation:
                m = {'role': msg.role, 'content': msg.content or ''}
                if msg.tool_call_id:
                    m['tool_call_id'] = msg.tool_call_id
                if msg.name:
                    m['name'] = msg.name
                if msg.tool_calls:
                    m['tool_calls'] = msg.tool_calls
                messages.append(m)

            tools = [
                {
                    'type': 'function',
                    'function': {
                        'name': t['name'],
                        'description': t['description'],
                        'parameters': t['parameters']
                    }
                }
                for t in self.tool_descriptions
            ]

            try:
                api_kwargs = self._build_openai_chat_kwargs(messages, tools)
                # Set hard wall-clock timeout via signal.alarm
                old_handler = signal.signal(signal.SIGALRM, _llm_timeout_handler)
                signal.alarm(LLM_WALL_CLOCK_TIMEOUT)
                try:
                    response = self.client.chat.completions.create(**api_kwargs)
                finally:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                self.total_turns += 1
                self._consecutive_errors = 0
                self.last_serving_model = str(getattr(response, 'model', None) or self.model)

                # Capture token usage (OpenAI chat completions format)
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                    self.last_output_tokens = getattr(usage, 'completion_tokens', 0) or 0
                    # Cache and reasoning details
                    # OpenAI 与 DeepSeek 的 Chat Completions 缓存字段位置不同。
                    self.last_cached_tokens = openai_chat_cached_tokens(usage)
                    ctd = getattr(usage, 'completion_tokens_details', None)
                    self.last_reasoning_tokens = getattr(ctd, 'reasoning_tokens', 0) or 0 if ctd else 0
                else:
                    self.last_input_tokens = 0
                    self.last_output_tokens = 0
                    self.last_cached_tokens = 0
                    self.last_reasoning_tokens = 0
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_cached_tokens += self.last_cached_tokens
                self.total_reasoning_tokens += self.last_reasoning_tokens

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=messages,
                        raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    )

                assistant_msg = response.choices[0].message

                # Log reasoning_content if present (e.g. GLM-5 reasoning model)
                reasoning_content = getattr(assistant_msg, 'reasoning_content', None)
                if not reasoning_content:
                    extras = getattr(assistant_msg, 'model_extra', {}) or {}
                    reasoning_content = extras.get('reasoning_content')
                if reasoning_content and self.tool_result_callback:
                    self.tool_result_callback(
                        self.total_turns, self.current_day, '_reasoning', {},
                        reasoning_content
                    )

                # Validate tool_call arguments are parseable JSON BEFORE appending
                # to conversation. Storing a tool_call with invalid-JSON args poisons
                # the history: some OpenAI-compat servers (e.g. Together) reject every
                # subsequent request with 400 "Input validation error" on replay.
                json_validation_error = None
                if assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        if tc.function.arguments:
                            try:
                                json.loads(tc.function.arguments)
                            except json.JSONDecodeError as je:
                                json_validation_error = (
                                    tc.function.name,
                                    str(je),
                                    tc.function.arguments[:300],
                                )
                                break

                if json_validation_error:
                    invalid_responses += 1
                    name, err, preview = json_validation_error
                    if invalid_responses >= self.max_invalid_responses_per_turn:
                        raise RuntimeError(
                            "Decision model repeatedly returned invalid tool arguments "
                            f"({invalid_responses} responses)"
                        )
                    print(f"  Invalid JSON in tool_call `{name}`: {err}. Feeding error back to LLM and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"Your previous response contained invalid JSON in the `{name}` tool_call arguments.\n"
                            f"JSON decode error: {err}\n"
                            f"Arguments started with: {preview}...\n\n"
                            f"Valid JSON escape sequences are limited to: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. "
                            f"Shell-style escapes like \\$ or \\! are NOT valid JSON. "
                            f"Please re-emit the tool call with valid JSON."
                        )
                    ))
                    continue

                tool_calls_data = None
                if assistant_msg.tool_calls:
                    tool_calls_data = []
                    for tc in assistant_msg.tool_calls:
                        tc_dict = {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            }
                        }
                        # Preserve Gemini thought_signature (required by Gemini
                        # OpenAI-compat endpoint — must be echoed back on replay).
                        tc_extras = getattr(tc, 'model_extra', None) or {}
                        extra_content = tc_extras.get('extra_content')
                        if extra_content:
                            tc_dict['extra_content'] = extra_content
                        tool_calls_data.append(tc_dict)

                self.conversation.append(Message(
                    role='assistant',
                    content=assistant_msg.content or '',
                    tool_calls=tool_calls_data
                ))

                if not assistant_msg.tool_calls:
                    invalid_responses += 1
                    if invalid_responses >= self.max_invalid_responses_per_turn:
                        raise RuntimeError(
                            "Decision model repeatedly returned no tool call "
                            f"({invalid_responses} responses)"
                        )
                    # LLM emitted no tool_call — feed feedback and retry.
                    print("  LLM returned no tool_call. Feeding feedback and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=self.NO_TOOL_FEEDBACK,
                    ))
                    continue

                # Handle tool calls — execute first, skip rest
                first_tc = assistant_msg.tool_calls[0]
                # Safe to parse — we already validated above.
                args = json.loads(first_tc.function.arguments) if first_tc.function.arguments else {}

                # Skip extra parallel tool calls
                for extra_tc in assistant_msg.tool_calls[1:]:
                    self.conversation.append(Message(
                        role='tool',
                        content=f"[Skipped - only one tool per turn. Call {extra_tc.function.name} again if needed.]",
                        tool_call_id=extra_tc.id,
                        name=extra_tc.function.name
                    ))

                self._pending_tool_calls = [{'id': first_tc.id, 'name': first_tc.function.name}]
                return Action(tool=first_tc.function.name, arguments=args)

            except (openai.OpenAIError, LLMTimeoutError):
                # SDK 已完成有限重试；继续外层重试会让本地服务故障时实验永久卡住。
                raise

    def _call_openai_responses(self) -> Optional[Action]:
        """Call the OpenAI Responses API, optionally enabling reasoning."""
        import signal
        import openai

        LLM_WALL_CLOCK_TIMEOUT = max(1, int(self.timeout_seconds))

        class LLMTimeoutError(Exception):
            pass

        def _llm_timeout_handler(signum, frame):
            raise LLMTimeoutError(f"LLM call exceeded {LLM_WALL_CLOCK_TIMEOUT}s wall-clock timeout")

        invalid_responses = 0
        while True:
            # Build input array from conversation
            input_items = []
            for msg in self.conversation:
                if msg.role == 'system':
                    continue  # System prompt goes in instructions parameter
                elif msg.role == 'user':
                    input_items.append({'role': 'user', 'content': msg.content or ''})
                elif msg.role == 'assistant':
                    if isinstance(msg.content, list):
                        # Raw response.output items from previous Responses API call
                        input_items.extend(msg.content)
                    else:
                        input_items.append({'role': 'assistant', 'content': msg.content or ''})
                elif msg.role == 'tool':
                    input_items.append({
                        'type': 'function_call_output',
                        'call_id': msg.tool_call_id,
                        'output': msg.content or '',
                    })

            # Build tools (Responses API format — no nested function wrapper)
            tools = [
                {
                    'type': 'function',
                    'name': t['name'],
                    'description': t['description'],
                    'parameters': t['parameters'],
                }
                for t in self.tool_descriptions
            ]

            try:
                api_kwargs = self._build_openai_responses_kwargs(input_items, tools)

                # Set hard wall-clock timeout via signal.alarm
                old_handler = signal.signal(signal.SIGALRM, _llm_timeout_handler)
                signal.alarm(LLM_WALL_CLOCK_TIMEOUT)
                try:
                    response = self.client.responses.create(**api_kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                self.total_turns += 1
                self._consecutive_errors = 0
                self.last_serving_model = str(getattr(response, 'model', None) or self.model)

                # Capture token usage (Responses API uses input_tokens/output_tokens)
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_input_tokens = getattr(usage, 'input_tokens', 0) or 0
                    self.last_output_tokens = getattr(usage, 'output_tokens', 0) or 0
                    # Cache and reasoning details
                    itd = getattr(usage, 'input_tokens_details', None)
                    self.last_cached_tokens = getattr(itd, 'cached_tokens', 0) or 0 if itd else 0
                    otd = getattr(usage, 'output_tokens_details', None)
                    self.last_reasoning_tokens = getattr(otd, 'reasoning_tokens', 0) or 0 if otd else 0
                else:
                    self.last_input_tokens = 0
                    self.last_output_tokens = 0
                    self.last_cached_tokens = 0
                    self.last_reasoning_tokens = 0
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_cached_tokens += self.last_cached_tokens
                self.total_reasoning_tokens += self.last_reasoning_tokens

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=input_items,
                        raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    )

                # Log reasoning content if present
                for item in response.output:
                    if getattr(item, 'type', '') == 'reasoning' and self.tool_result_callback:
                        reasoning_text = ''
                        for summary in getattr(item, 'summary', []) or []:
                            reasoning_text += getattr(summary, 'text', '') + '\n'
                        if reasoning_text.strip():
                            self.tool_result_callback(
                                self.total_turns, self.current_day, '_reasoning', {},
                                reasoning_text.strip()
                            )

                # Find function_call items
                function_calls = [item for item in response.output
                                  if getattr(item, 'type', '') == 'function_call']

                # Validate each function_call's arguments JSON BEFORE storing. An
                # invalid-JSON tool_call poisons the conversation (server-side
                # validators reject every subsequent request on replay).
                json_validation_error = None
                for fc in function_calls:
                    if fc.arguments:
                        try:
                            json.loads(fc.arguments)
                        except json.JSONDecodeError as je:
                            json_validation_error = (
                                fc.name,
                                str(je),
                                fc.arguments[:300],
                            )
                            break

                if json_validation_error:
                    invalid_responses += 1
                    name, err, preview = json_validation_error
                    if invalid_responses >= self.max_invalid_responses_per_turn:
                        raise RuntimeError(
                            "Decision model repeatedly returned invalid tool arguments "
                            f"({invalid_responses} responses)"
                        )
                    print(f"  Invalid JSON in function_call `{name}`: {err}. Feeding error back to LLM and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"Your previous response contained invalid JSON in the `{name}` tool_call arguments.\n"
                            f"JSON decode error: {err}\n"
                            f"Arguments started with: {preview}...\n\n"
                            f"Valid JSON escape sequences are limited to: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. "
                            f"Shell-style escapes like \\$ or \\! are NOT valid JSON. "
                            f"Please re-emit the tool call with valid JSON."
                        )
                    ))
                    continue

                # Store raw output items for conversation history reconstruction
                self.conversation.append(Message(
                    role='assistant',
                    content=list(response.output),
                ))

                if not function_calls:
                    invalid_responses += 1
                    if invalid_responses >= self.max_invalid_responses_per_turn:
                        raise RuntimeError(
                            "Decision model repeatedly returned no tool call "
                            f"({invalid_responses} responses)"
                        )
                    # LLM emitted no tool_call — feed feedback and retry.
                    print("  LLM returned no function_call. Feeding feedback and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=self.NO_TOOL_FEEDBACK,
                    ))
                    continue

                # Handle tool calls — execute first, skip rest
                first_fc = function_calls[0]
                # Safe to parse — we already validated above.
                args = json.loads(first_fc.arguments) if first_fc.arguments else {}

                # Skip extra parallel tool calls
                for extra_fc in function_calls[1:]:
                    self.conversation.append(Message(
                        role='tool',
                        content=f"[Skipped - only one tool per turn. Call {extra_fc.name} again if needed.]",
                        tool_call_id=extra_fc.call_id,
                        name=extra_fc.name
                    ))

                self._pending_tool_calls = [{'id': first_fc.call_id, 'name': first_fc.name}]
                return Action(tool=first_fc.name, arguments=args)

            except (openai.OpenAIError, LLMTimeoutError):
                # SDK 已完成有限重试；继续外层重试会让本地服务故障时实验永久卡住。
                raise

    def _anthropic_content_text(self, content: Any) -> str:
        """Best-effort text extraction from Anthropic content blocks."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get('text') or block.get('content')
            else:
                text = getattr(block, 'text', None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)

    def _anthropic_response_dict(self, response: Any) -> Dict[str, Any]:
        """Convert an Anthropic response object to a JSON-safe dict."""
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_none=False, by_alias=True)
        if isinstance(response, dict):
            return response
        return {}

    def _record_anthropic_response_metadata(self, response: Any) -> None:
        """Track the model actually returned by Anthropic for logging."""
        response_dict = self._anthropic_response_dict(response)
        self.last_serving_model = str(response_dict.get('model') or self.model)

    def _anthropic_no_tool_feedback(self, response: Any, assistant_content: Any) -> str:
        """Feedback used when Anthropic returns text instead of a tool."""
        preview = self._anthropic_content_text(assistant_content).strip()
        if len(preview) > 1200:
            preview = preview[:1200] + "..."

        return (
            f"{self.NO_TOOL_FEEDBACK} "
            f"Previous non-tool response preview: {preview or '(no text)'}"
        )

    def _call_anthropic(self) -> Optional[Action]:
        """Call Anthropic/Bedrock API and parse the response."""
        import anthropic
        import copy

        no_tool_retries = 0

        while True:
            messages = []
            for msg in self.conversation:
                if msg.role == 'system':
                    continue
                messages.append({'role': msg.role, 'content': copy.deepcopy(msg.content)})

            # 价格模型尚未表达 Anthropic 缓存写入及其 TTL 价格，
            # 因此默认路径不主动创建缓存断点。恢复的历史中若存在
            # cache_control，也必须移除，避免断点续跑时意外产生未计价写入。
            def _strip_cache_control(content):
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and 'cache_control' in block:
                            del block['cache_control']

            for msg in messages:
                _strip_cache_control(msg.get('content'))

            system_text = self._get_system_prompt_with_memory()
            system_content = system_text

            from .tools import get_bash_agent_anthropic_tools
            tools = get_bash_agent_anthropic_tools()

            api_kwargs = {
                'model': self.model,
                'max_tokens': self.max_output_tokens,
                'system': system_content,
                'messages': messages,
                'tools': tools,
            }
            if self.temperature is not None:
                api_kwargs['temperature'] = self.temperature
            if self.top_p is not None:
                api_kwargs['top_p'] = self.top_p
            api_kwargs.update(self.request_options)
            anthropic_messages = self.client.messages

            # Anthropic SDK refuses non-streaming when max_tokens implies > 10min
            # budget (raised in _calculate_nonstreaming_timeout). Always stream
            # for max_tokens > 64000.
            use_streaming = api_kwargs['max_tokens'] > 64000
            if 'thinking' in api_kwargs:
                use_streaming = True

            try:
                if use_streaming:
                    with anthropic_messages.stream(**api_kwargs) as stream:
                        response = stream.get_final_message()
                else:
                    response = anthropic_messages.create(**api_kwargs)

                self.total_turns += 1
                self._consecutive_errors = 0
                self._record_anthropic_response_metadata(response)

                # Capture token usage (Anthropic format)
                usage = getattr(response, 'usage', None)
                if usage:
                    uncached_input_tokens = getattr(usage, 'input_tokens', 0) or 0
                    self.last_output_tokens = getattr(usage, 'output_tokens', 0) or 0
                    # Anthropic 的 input_tokens 不含缓存读写，统一归一为总输入量。
                    self.last_cached_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0
                    cache_creation_tokens = (
                        getattr(usage, 'cache_creation_input_tokens', 0) or 0
                    )
                    # TODO: Anthropic 缓存写入需要独立价格。计价模型支持前直接
                    # 中止，避免将其误算成普通输入并污染实验成本。
                    if cache_creation_tokens:
                        raise NotImplementedError(
                            "Anthropic cache creation pricing is not configured; "
                            "disable prompt-cache writes or extend the pricing model"
                        )
                    self.last_input_tokens = (
                        uncached_input_tokens
                        + self.last_cached_tokens
                    )
                    output_details = getattr(usage, 'output_tokens_details', None)
                    self.last_reasoning_tokens = (
                        getattr(output_details, 'thinking_tokens', 0) or 0
                        if output_details else 0
                    )
                else:
                    self.last_input_tokens = 0
                    self.last_output_tokens = 0
                    self.last_cached_tokens = 0
                    self.last_reasoning_tokens = 0
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_cached_tokens += self.last_cached_tokens
                self.total_reasoning_tokens += self.last_reasoning_tokens

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=messages,
                        raw_response=self._anthropic_response_dict(response) or str(response),
                    )

                assistant_content = response.content
                self.conversation.append(Message(
                    role='assistant',
                    content=assistant_content
                ))

                tool_use_blocks = [block for block in assistant_content if block.type == 'tool_use']
                if not tool_use_blocks:
                    no_tool_retries += 1
                    stop_reason = getattr(response, 'stop_reason', '') or 'no_tool_use'
                    if self.tool_result_callback:
                        self.tool_result_callback(
                            self.total_turns,
                            self.current_day,
                            '_anthropic_no_tool',
                            {'stop_reason': stop_reason, 'attempt': no_tool_retries},
                            self._anthropic_content_text(assistant_content),
                        )
                    if no_tool_retries >= self.max_invalid_responses_per_turn:
                        raise RuntimeError(
                            "Anthropic response did not include a tool_use block after "
                            f"{no_tool_retries} attempts (last stop_reason={stop_reason!r})."
                        )
                    print(
                        f"  Anthropic returned no tool_use "
                        f"(stop_reason={stop_reason!r}); feeding feedback and regenerating."
                    )
                    self.conversation.append(Message(
                        role='user',
                        content=self._anthropic_no_tool_feedback(response, assistant_content),
                    ))
                    continue

                first_tool = tool_use_blocks[0]

                # Skip extra parallel tool calls
                partial_results = []
                for extra in tool_use_blocks[1:]:
                    partial_results.append({
                        'type': 'tool_result',
                        'tool_use_id': extra.id,
                        'content': f"[Skipped - only one tool per turn. Call {extra.name} again if needed.]",
                    })

                self._pending_tool_calls = [{'id': first_tool.id, 'name': first_tool.name, '_partial_results': partial_results}]
                return Action(tool=first_tool.name, arguments=first_tool.input or {})

            except anthropic.APIError as e:
                # 只重试 Anthropic SDK 明确报告的 Provider 异常。
                import traceback
                status = getattr(e, 'status_code', 0) or 0
                is_retryable = isinstance(
                    e, (anthropic.APIConnectionError, anthropic.APITimeoutError)
                ) or (
                    isinstance(e, anthropic.APIStatusError)
                    and (status == 429 or status >= 500)
                )
                if not is_retryable:
                    raise
                if str(e).startswith("Anthropic response did not include a tool_use block"):
                    raise
                error_msg = f"Anthropic LLM call error: {e}"
                tb = traceback.format_exc()
                print(f"\n{'='*60}")
                print(f"ERROR in BashAgent._call_anthropic()")
                print(f"{'='*60}")
                print(error_msg)
                print(f"Traceback:\n{tb}")
                print(f"{'='*60}\n")

                self._consecutive_errors += 1
                if self._consecutive_errors <= 3:
                    wait = 2 ** self._consecutive_errors
                    print(f"  Retrying in {wait}s (attempt {self._consecutive_errors}/3)...")
                    time.sleep(wait)
                    return self._call_anthropic()

                raise RuntimeError(
                    f"LLM failed {self._consecutive_errors} consecutive times. "
                    f"Last error: {e}"
                ) from e
