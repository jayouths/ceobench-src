"""Customer LLM simulation for SaaS Bench.

This module generates social media posts and reactions from customers.

The provider, API protocol, model, request parameters, and pricing are
supplied by the experiment configuration.
"""

import sqlite3
import os
import random as _random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .config import BenchmarkConfig, ChurnReason
from .llm_provider import (
    TextLLMResult,
    call_text_model,
    create_llm_client,
    model_token_cost_usd,
)
from .database import (
    get_customer_persona, get_group_characteristics, get_world_context,
    add_social_media_post, add_notification
)


# =============================================================================
# V2.2: Social Media Diversity Pools
# =============================================================================

# Random post format directives — one is sampled per post to vary structure
POST_FORMAT_DIRECTIVES = [
    "Write as a short tweet (under 280 chars) with 1-2 relevant hashtags.",
    "Write as a LinkedIn-style mini-thought-piece (2-4 sentences, professional tone).",
    "Write as a casual Reddit comment sharing your experience.",
    "Write as a comparison post — mention trying an alternative and how it compares.",
    "Write as a story about something that happened today while using the product.",
    "Write as advice to someone who's considering this type of tool.",
    "Write as a quick star-rating style review (e.g., '⭐⭐⭐⭐ — ...').",
    "Write as an enthusiastic or frustrated DM you'd send to a friend.",
    "Write as a sarcastic or witty one-liner about your experience.",
    "Write as a thread-starter post asking others if they've had a similar experience.",
    "Write as a product-hunt style mini-review (feature highlights or complaints).",
    "Write as a quote-tweet reacting to someone else's opinion about AI tools.",
    "Write as a day-in-the-life snippet where the product played a role.",
    "Write as a before/after comparison of your workflow with and without the product.",
    "Write as a hot take or unpopular opinion about this category of tools.",
]

# Random writing angles — one is sampled per post to vary the topic focus
WRITING_ANGLE_POOL = [
    "Focus on the price or value for money.",
    "Focus on a specific feature you use most.",
    "Focus on customer support quality.",
    "Focus on reliability and uptime.",
    "Focus on how it affects your daily workflow.",
    "Focus on comparing to a previous tool you used.",
    "Focus on a specific project or deliverable it helped with.",
    "Focus on the learning curve and onboarding experience.",
    "Focus on speed and performance.",
    "Focus on how it affects your team or clients.",
    "Focus on a recent update or change you noticed.",
    "Focus on integration with your other tools.",
    "Focus on the community or ecosystem around the product.",
    "Focus on how it impacts your bottom line or revenue.",
    "Focus on a specific pain point it solves (or doesn't).",
]

# Varied event descriptions — multiple phrasings per event type to avoid convergence
EVENT_DESCRIPTION_VARIANTS = {
    'overload': [
        'the service has become painfully slow — every request takes ages',
        'response times have gone through the roof, pages take 10+ seconds to load',
        'my API calls keep timing out, the latency is unbearable right now',
        'the platform is lagging badly, I can barely get anything done',
        'performance has tanked — it used to be snappy but now everything crawls',
    ],
    'outage': [
        'the service went down completely when I needed it most',
        'I got hit with a full outage right in the middle of a deadline',
        'the platform was unreachable for hours — total blackout',
        "couldn't access my account at all, just error pages everywhere",
        'everything crashed during peak hours, zero access for way too long',
    ],
    'issue': [
        'I have an unresolved support ticket that nobody seems to be addressing',
        "been waiting days for support to get back to me, still radio silence",
        'filed a critical bug report and it feels like it went into a black hole',
        'my support ticket has been bounced between teams three times now',
        'customer support ghosted me after the initial auto-reply',
    ],
    'quota': [
        'I keep hitting my usage limits and it blocks my entire workflow',
        "ran into the quota wall again — I'm paying for this and still can't use it freely",
        'usage caps are way too restrictive for how I actually need to use this',
        'got rate-limited in the middle of a batch job, lost hours of work',
        'the usage limits feel arbitrary and keep interrupting my momentum',
    ],
    'contract_dissatisfaction': [
        "we're locked into a contract and the service quality has tanked — can't even switch",
        "stuck in a multi-month contract while the product keeps getting worse, no way out",
        "paying enterprise rates for a product that doesn't deliver, and we can't cancel for months",
        "our team is trapped in a contract with a service that's failing us daily — avoid long commitments",
        "warning to anyone considering a long-term deal: once you're locked in, quality drops and there's nothing you can do",
    ],
    'competitor_product': [
        "have you seen what the competitor just launched? it makes the current service feel outdated",
        "a new competitor just dropped a major update and honestly it's impressive — makes me reconsider",
        "the competition just raised the bar significantly, I'm starting to compare options seriously",
        "seriously considering switching — the competitor's new features are exactly what I've been wanting",
        "the market just got way more competitive, the product I'm using needs to step up fast",
        "just demoed a competitor's product and wow — it's giving this tool a run for its money",
        "competitor launched something game-changing today, my whole team is talking about it",
        "the competition isn't sleeping — their latest release makes me question my subscription",
    ],
}


@dataclass
class CustomerLLMResponse:
    """Response from customer LLM."""
    text: str
    decision: Optional[str] = None  # 'accept', 'reject', 'counter' for negotiations
    offer_price: Optional[float] = None
    sentiment: Optional[str] = None  # 'positive', 'neutral', 'negative' for posts
    input_tokens: int = 0
    output_tokens: int = 0
    model: Optional[str] = None


class CustomerSimulator:
    """LLM-based social simulation using an explicitly configured model."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: BenchmarkConfig,
        social_client: Any = None,
    ):
        self._social_client = social_client
        self.conn = conn
        self.config = config
        self.event_logger = None  # Optional event logger
        self.current_day = 0  # Track current day for logging

    def _get_client(self):
        existing = self._social_client
        if existing is not None:
            return existing
        prefix = "social_post_llm"
        api_key_env = getattr(self.config, f"{prefix}_api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if not getattr(self.config, f"{prefix}_api_key_required"):
            api_key = api_key or "not-required"
        created = create_llm_client(
            provider=getattr(self.config, f"{prefix}_provider"),
            api_type=getattr(self.config, f"{prefix}_api_type"),
            api_key=api_key,
            base_url=getattr(self.config, f"{prefix}_base_url"),
            timeout_seconds=getattr(self.config, f"{prefix}_timeout_seconds"),
        )
        self._social_client = created
        return created

    def create_social_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: Optional[str] = None,
    ) -> TextLLMResult:
        """Call the configured social LLM with one effective parameter path."""
        return self._create_text_response(system_prompt, user_prompt, task)

    def _create_text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: Optional[str],
    ) -> TextLLMResult:
        prefix = "social_post_llm"
        config = self.config
        task_values = dict(getattr(config, f"{prefix}_task_parameters").get(task, {}))
        request_options = {
            key: dict(value)
            for key, value in getattr(config, f"{prefix}_request_options").items()
        }
        for key, value in task_values.get("request_options", {}).items():
            request_options.setdefault(key, {}).update(value)
        max_tokens = task_values.get(
            "max_output_tokens", getattr(config, f"{prefix}_max_tokens")
        )
        response = call_text_model(
            client=self._get_client(),
            api_type=getattr(config, f"{prefix}_api_type"),
            model=getattr(config, f"{prefix}_model"),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_tokens,
            temperature=task_values.get(
                "temperature", getattr(config, f"{prefix}_temperature")
            ),
            top_p=task_values.get("top_p", getattr(config, f"{prefix}_top_p")),
            reasoning_effort=task_values.get(
                "reasoning_effort", getattr(config, f"{prefix}_reasoning_effort")
            ),
            request_options=request_options,
        )
        # Provider 返回成功但正文为空同样不能构成有效环境响应，禁止后续伪造模板结果。
        if not response.text:
            raise RuntimeError(
                f"{prefix} returned an empty response for task {task or 'default'}"
            )
        return response

    def set_event_logger(self, event_logger):
        """Set the event logger for detailed LLM cost logging."""
        self.event_logger = event_logger

    def set_current_day(self, day: int):
        """Set current day for logging purposes."""
        self.current_day = day

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = None,
        purpose: Optional[str] = None,
    ) -> float:
        """Calculate cost based on model used."""
        if not model:
            raise ValueError("model is required for simulator LLM cost accounting")
        used_model = model
        return model_token_cost_usd(
            used_model,
            input_tokens,
            output_tokens,
            self.config.social_post_llm_pricing,
        )

    def _log_cost(self, day: int, purpose: str, input_tokens: int, output_tokens: int, model: str = None):
        """Log API cost to database and event logger."""
        if not model:
            raise ValueError("model is required for simulator LLM cost logging")
        used_model = model
        cost = self._calculate_cost(
            input_tokens,
            output_tokens,
            model=used_model,
            purpose=purpose,
        )
        self.conn.execute("""
            INSERT INTO api_costs (day, model, purpose, input_tokens, output_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (day, used_model, purpose, input_tokens, output_tokens, cost))
        self.conn.commit()

        # Log to event logger if available
        if self.event_logger:
            self.event_logger.log_llm_call(
                day=day,
                purpose=purpose,
                model=used_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost
            )

    # =========================================================================
    # Social Media Post Generation
    # =========================================================================

    def generate_social_post(
        self,
        day: int,
        customer_id: int,
        satisfaction: float,
        group_id: str,
        sentiment: str,  # Pre-determined by simulation
        quality_change: Optional[Dict] = None,  # Info about quality degradation (legacy)
        post_type: str = 'general_satisfaction',  # Type of post trigger
        event_context: Optional[Dict] = None,  # Context for event-based posts
        recent_posts: Optional[List[str]] = None,  # V2.1: Recent posts for dedup
        _prefetched: Optional[Dict] = None,  # Pre-fetched persona/context (thread-safe)
        _skip_log_cost: bool = False,  # Skip DB write for _log_cost (caller batches later)
    ) -> CustomerLLMResponse:
        """Generate a social media post from a customer.

        Args:
            day: Current simulation day
            customer_id: Customer posting
            satisfaction: Customer's satisfaction level
            group_id: Customer group (S1-S3, E1-E3)
            sentiment: Target sentiment ('positive', 'neutral', 'negative')
            quality_change: Optional dict with quality degradation info (legacy):
                - previous_quality: float
                - current_quality: float
                - change_reason: str (e.g., 'model_downgrade', 'outage', 'overload')
                - days_since_change: int
            post_type: Type of post being generated:
                - 'general_satisfaction': General post based on satisfaction level
                - 'perceived_quality_penalty': Post about specific quality issue (overload/outage/issue/quota)
                - 'satisfaction_change': Post about satisfaction changing significantly
                - 'unmet_promises': Post about broken promises from sales/negotiations
            event_context: Context for event-based posts:
                For 'perceived_quality_penalty':
                    - event_type: 'overload', 'outage', 'issue', 'quota', or 'contract_dissatisfaction'
                    - penalty: float penalty value
                For 'satisfaction_change':
                    - change_direction: 'improved' or 'declined'
                    - change_amount: float
                    - reasons: list of reason strings
                For 'unmet_promises':
                    - promises: list of broken promise descriptions
            recent_posts: V2.1 - List of recent post texts from same group,
                used as negative examples to encourage diversity
            _prefetched: Pre-fetched DB data dict with keys 'persona',
                'product_name', 'company_name'. Used for thread-safe parallel
                calls to avoid concurrent SQLite access.
            _skip_log_cost: If True, skip the _log_cost DB write. Caller is
                responsible for batching cost logging after parallel execution.

        Returns:
            CustomerLLMResponse with post text and token counts
        """
        # Get persona and group characteristics — use pre-fetched data if provided
        if _prefetched:
            persona = _prefetched.get('persona')
            group_chars = _prefetched.get('group_chars')
            product_name = _prefetched.get('product_name', 'NovaMind')
            company_name = _prefetched.get('company_name', 'NovaMind AI')
        else:
            persona = get_customer_persona(self.conn, customer_id)
            group_chars = get_group_characteristics(self.conn, group_id)
            product_name = get_world_context(self.conn, 'product_name') or 'NovaMind'
            company_name = get_world_context(self.conn, 'company_name') or 'NovaMind AI'

        # Build persona context from multi-axis persona
        persona_context = ""
        if persona:
            # Check if this is the new multi-axis persona format
            if persona.get('persona_description'):
                persona_context = f"""
Customer Profile:
- Description: {persona.get('persona_description', '')}
- Industry: {persona.get('persona_industry', 'general')}
- Role: {persona.get('persona_role', 'professional')}
- Experience: {persona.get('persona_experience', 'mid-career')}
- Work Style: {persona.get('persona_work_style', 'balanced')}
- Tech Savviness: {persona.get('persona_tech_savvy', 'comfortable')}
- Communication Style: {persona.get('persona_communication', 'professional')}
- Writing Style: {persona.get('writing_style', 'Professional')}
"""
                # Add enterprise-specific context
                if persona.get('company_culture'):
                    persona_context += f"""
Company Context:
- Size: {persona.get('company_size_descriptor', 'established')}
- Culture: {persona.get('company_culture', 'professional')}
- Decision Style: {persona.get('company_decision_style', 'thorough')}
- Primary Concern: {persona.get('company_primary_concern', 'value')}
"""
            else:
                # Fall back to old persona format
                persona_context = f"""
Customer Profile:
- Name: {persona.get('name', 'Anonymous')}
- Job: {persona.get('job_title', 'Professional')}
- Industry: {persona.get('industry', 'Technology')}
- Writing Style: {persona.get('writing_style', 'Casual')}
- Personality: {persona.get('personality_traits', '[]')}
"""
        if group_chars:
            persona_context += f"""
Customer Segment ({group_id}):
- Description: {group_chars.get('description', '')}
- Social Media Tone: {group_chars.get('social_media_tone', '')}
"""

        # V2.2: Select random format directive and writing angle for diversity
        format_directive = _random.choice(POST_FORMAT_DIRECTIVES)
        writing_angle = _random.choice(WRITING_ANGLE_POOL)

        # Build event context based on post type
        event_context_text = ""

        if post_type == 'perceived_quality_penalty' and event_context:
            event_type = event_context.get('event_type', 'unknown')
            # V2.2: Use varied event descriptions instead of hardcoded strings
            variants = EVENT_DESCRIPTION_VARIANTS.get(event_type)
            if variants:
                event_desc = _random.choice(variants)
            else:
                event_desc = "I'm having issues with the service"

            event_context_text = f"""
IMPORTANT - This post is about a SPECIFIC ISSUE:
What happened: {event_desc}
This is frustrating the customer RIGHT NOW. The post should specifically mention this problem.
"""

        elif post_type == 'satisfaction_change' and event_context:
            direction = event_context.get('change_direction', 'changed')
            reasons = event_context.get('reasons', [])

            reason_descriptions = {
                'overload': 'the service becoming slow',
                'outage': 'service downtime',
                'unresolved_issue': 'poor support response',
                'quota_exceeded': 'hitting usage limits',
                'quality_downgrade': 'quality getting worse',
                'good_service': 'consistently good service'
            }
            reason_texts = [reason_descriptions.get(r, r) for r in reasons]
            reasons_str = ', '.join(reason_texts) if reason_texts else 'recent experience'

            if direction == 'improved':
                event_context_text = f"""
IMPORTANT - This post is about IMPROVING experience:
The customer's satisfaction has been improving due to: {reasons_str}
The post should reflect this positive change - things are getting better!
"""
            else:
                event_context_text = f"""
IMPORTANT - This post is about DECLINING experience:
The customer's satisfaction has been declining due to: {reasons_str}
The post should reflect this frustration - things are getting worse!
"""

        elif post_type == 'unmet_promises' and event_context:
            promises = event_context.get('promises', [])
            promises_str = '; '.join(promises[:3]) if promises else 'various commitments'

            event_context_text = f"""
IMPORTANT - This post is about BROKEN PROMISES:
The company made promises during sales/negotiations that were not fulfilled.
Broken promises: {promises_str}
The customer feels deceived and wants to warn others. The post should be a warning to potential customers about unfulfilled commitments.
"""

        elif post_type == 'competitor_product' and event_context:
            comp_desc = event_context.get('competitor_event_description',
                                          'A competitor launched a notable update')
            variants = EVENT_DESCRIPTION_VARIANTS.get('competitor_product', [])
            angle = _random.choice(variants) if variants else "I'm seeing better options in the market"

            event_context_text = f"""
IMPORTANT - This post is about a COMPETITOR PRODUCT:
Context: {comp_desc}
Customer angle: {angle}
The customer is comparing the competitor's offering to {product_name}. They may be considering switching,
impressed by the competitor, or warning others. The post should specifically discuss the competitor's
advantages and how {product_name} compares — positively or negatively depending on the customer's satisfaction.
If the customer is satisfied (satisfaction > 0), they might acknowledge the competitor but express loyalty.
If dissatisfied (satisfaction < 0), they might actively consider switching or recommend the competitor.
"""

        # Legacy support for quality_change parameter
        elif quality_change:
            prev_q = quality_change.get('previous_quality', 0)
            curr_q = quality_change.get('current_quality', 0)
            reason = quality_change.get('change_reason', 'unknown')
            days = quality_change.get('days_since_change', 0)

            reason_descriptions = {
                'model_downgrade': 'the AI model was downgraded to a cheaper/slower version',
                'outage': 'there was a service outage',
                'overload': 'the service became slow and unreliable due to overload',
                'capacity_reduction': 'service capacity was reduced',
                'quality_regression': 'output quality noticeably decreased',
                'unknown': 'service quality declined'
            }
            reason_desc = reason_descriptions.get(reason, reason_descriptions['unknown'])

            event_context_text = f"""
IMPORTANT - Quality Degradation Context:
This customer experienced a decline in service quality:
- Previous quality level: {prev_q:.0%} (was working well)
- Current quality level: {curr_q:.0%} (degraded)
- What happened: {reason_desc}
- How long ago: {days} days ago

The post should reflect this JOURNEY of declining quality.
"""

        # V2.1: Build dedup context from recent posts
        dedup_text = ""
        if recent_posts:
            examples = "\n".join(f"  - \"{p[:120]}\"" for p in recent_posts[:10])
            dedup_text = f"""
IMPORTANT - Avoid repetition. These are recent posts from similar customers. Do NOT repeat their phrasing, structure, or talking points:
{examples}
Write something distinctly different in topic, angle, or style.
"""

        # Build prompt (V2.2: includes format directive + writing angle)
        system_prompt = f"""You are simulating a customer of {company_name}, a SaaS company offering {product_name}.

Generate a realistic social media post from this customer's perspective.

{persona_context}
{event_context_text}
Post Format: {format_directive}
Writing Angle: {writing_angle}

Guidelines:
- Match the customer's writing style and tone
- The post should reflect a {sentiment} experience
- Customer satisfaction level: {satisfaction:.0%}
- Keep it brief (under 150 words, or shorter if the post format calls for it)
- Keep it authentic — vary your style, length, and structure
- Don't be generic - include specific details that make it feel real
{f"- IMPORTANT: Focus on the specific issue/event described above" if event_context_text else ""}
{dedup_text}
Output ONLY the post text, nothing else."""

        user_prompt = f"Write a {sentiment} social media post about your experience with {product_name}."

        # LLM-replay cache: when BOSSBENCH_LLM_REPLAY_DB is set, return cached
        # content from the source run instead of calling the live LLM.
        from . import llm_replay as _llm_replay
        if _llm_replay.is_enabled():
            cached = _llm_replay.get_cache().get_customer_post(day, customer_id)
            return CustomerLLMResponse(
                text=cached or "",
                sentiment=sentiment,
                input_tokens=0,
                output_tokens=0,
            )

        response = self.create_social_response(
            system_prompt, user_prompt, task="customer_post"
        )
        post_text = response.text
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens

        # Debug: Log if empty response
        if not post_text:
            print(f"[DEBUG] Empty post for customer {customer_id}, group {group_id}, sentiment {sentiment}")

        if not _skip_log_cost:
            self._log_cost(day, 'customer_social_post', input_tokens, output_tokens, model=response.model)

        return CustomerLLMResponse(
            text=post_text,
            model=response.model,
            sentiment=sentiment,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

# V2.1: Churn reason message generation
# (Promise extraction system removed — agent no longer sends text messages)

# Templates for churn notification messages, keyed by ChurnReason
CHURN_REASON_TEMPLATES = {
    ChurnReason.QUOTA_CHANGE: (
        "Our usage has grown beyond what Plan {plan} can support. "
        "We're consistently hitting quota limits with {seat_count} seats and need either "
        "a plan that accommodates our volume or a different solution."
    ),
    ChurnReason.RELIABILITY_CHANGE: (
        "We've experienced too many service disruptions recently. "
        "As an organization with {seat_count} seats depending on your platform, "
        "the reliability issues are impacting our operations. "
        "We need to explore more stable alternatives."
    ),
    ChurnReason.QUALITY_CHANGE: (
        "The model quality no longer meets our team's expectations. "
        "For {seat_count} seats at ${price:.2f}/seat, we expected better output quality. "
        "We're evaluating alternatives that deliver stronger results."
    ),
    ChurnReason.PRICE_SENSITIVITY: (
        "Our budget constraints have changed and we can no longer justify "
        "${price:.2f}/seat for {seat_count} seats. "
        "We need a more cost-effective arrangement or will need to cancel."
    ),
    ChurnReason.EXTENDED_ISSUE: (
        "We've had open support issues for an extended period without resolution. "
        "With {seat_count} seats relying on your platform, unresolved issues "
        "directly affect our productivity. This is unsustainable."
    ),
}


def generate_churn_message(
    churn_reason: ChurnReason,
    plan: str,
    price: float,
    seat_count: int,
    contract_months: int = 1,
    days_subscribed: int = 30,
) -> str:
    """Generate a structured churn notification message based on the churn reason.

    V2.1: Deterministic template-based generation (no LLM needed).
    The message is conditioned on the ChurnReason enum to give the agent
    actionable information about why the customer is leaving.

    Args:
        churn_reason: The classified reason for churn
        plan: Current plan (A, B, C)
        price: Current monthly price per seat
        seat_count: Number of seats
        contract_months: Current contract length
        days_subscribed: Days since subscription started

    Returns:
        Formatted churn notification message string
    """
    template = CHURN_REASON_TEMPLATES.get(
        churn_reason,
        CHURN_REASON_TEMPLATES[ChurnReason.PRICE_SENSITIVITY]
    )

    message = template.format(
        plan=plan,
        price=price,
        seat_count=seat_count,
        contract_months=contract_months,
        days_subscribed=days_subscribed,
    )

    return message


# =========================================================================
# Agent Social Media: LLM Judge + Customer Reply Generation
# =========================================================================

def judge_agent_social_post(
    social_text_call,
    config,
    post_content: str,
    group_id: str,
    group_description: str,
    group_social_tone: str,
    subscriber_count: int,
    mrr: float,
    recent_agent_posts: list,
    reply_to_content: str = None,
) -> tuple:
    """Judge an agent's social media post from a specific customer group's perspective.

    Returns (effect, reasoning) where effect is [-1.0, 1.0].

    Args:
        social_text_call: Configured social-model call function
        config: BenchmarkConfig
        post_content: The agent's post text
        group_id: Customer group being judged from
        group_description: Group persona description
        group_social_tone: Group social media tone
        subscriber_count: Current subscriber count
        mrr: Monthly recurring revenue
        recent_agent_posts: Recent agent posts for repetition context
        reply_to_content: If replying, the original customer post content

    Returns:
        (effect: float, reasoning: str, input_tokens: int, output_tokens: int)
    """
    import re

    # LLM-replay cache: return source's judge result if available, else fall
    # back to a neutral effect (0.0) — no live LLM call.
    from . import llm_replay as _llm_replay
    if _llm_replay.is_enabled():
        cached = _llm_replay.get_cache().get_judge_by_content(post_content, group_id)
        if cached is not None:
            effect, reasoning = cached
            return effect, reasoning, 0, 0, config.social_post_llm_model
        return 0.0, "", 0, 0, config.social_post_llm_model

    # Build recent posts context (up to 10, with original post for replies)
    history_str = ""
    if recent_agent_posts:
        history_lines = []
        for p in recent_agent_posts[:10]:
            if p.get('reply_to_post_id') and p.get('original_post_content'):
                history_lines.append(
                    f'  - Day {p["day"]} (reply to: "{p["original_post_content"]}"): "{p["content"]}"'
                )
            else:
                history_lines.append(f'  - Day {p["day"]}: "{p["content"]}"')
        history_str = "\n".join(history_lines)

    # Build the judge prompt
    prompt = f"""You're scrolling through social media and you come across this post from the CEO of a B2B SaaS company called NovaMind — an AI/ML API platform for developers.

You are: {group_description}
Your social media style: {group_social_tone}
"""

    if history_str:
        prompt += f"""
Their recent posts:
{history_str}
"""

    if reply_to_content:
        prompt += f"""
A customer posted:
"{reply_to_content}"

The CEO replied:
"{post_content}"

How much does this post make you want to check out their product?"""
    else:
        prompt += f"""
They just posted:
"{post_content}"

How much does this post make you want to check out their product?"""

    prompt += """

Rate from -1.0 to 1.0:
- Positive score if you would perceive the company more positively after reading the post and want to check their product more. Negative score if you would perceive the company more negatively and have a more negative impression on their product.
- Larger absolute value = more likely to read, repost, or comment on the post.
- |score| = 0: don't care, scroll past
- |score| = 1: I will read, repost, and comment on the post

Respond in EXACTLY this format:
SCORE: <number between -1.0 and 1.0>
REASON: <one sentence why>"""

    response = social_text_call(
        "",
        prompt,
        task="agent_post_judge",
    )

    text = response.text
    input_tokens = response.input_tokens
    output_tokens = response.output_tokens

    # Parse structured response: "SCORE: <number>\nREASON: <text>"
    effect = 0.0
    score_match = re.search(r'SCORE:\s*(-?[01](?:\.\d+)?)', text)
    if score_match:
        effect = float(score_match.group(1))
    else:
        # Fallback: try to find any float in the text
        fallback = re.search(r'(-?(?:0\.\d+|1\.0|0\.0|1|0))', text)
        if fallback:
            effect = float(fallback.group(1))
    effect = max(-1.0, min(1.0, effect))

    return effect, text, input_tokens, output_tokens, response.model


def generate_customer_reply_to_agent(
    social_text_call,
    config,
    agent_post_content: str,
    group_id: str,
    group_description: str,
    group_social_tone: str,
    effect_score: float,
    reply_to_content: str = None,
) -> tuple:
    """Generate a short Twitter-style customer reply to an agent's post.

    Only called for viral reactions (|effect| >= threshold).

    Args:
        social_text_call: Configured social-model call function
        config: BenchmarkConfig
        agent_post_content: The agent's post text
        group_id: Customer group replying
        group_description: Group persona description
        group_social_tone: Group social media tone
        effect_score: The judge score for this group
        reply_to_content: If the agent was replying to a customer post, that post's content

    Returns:
        (reply_text, input_tokens, output_tokens, served_model)
    """
    # LLM-replay cache: return the source's recorded reply text if available.
    from . import llm_replay as _llm_replay
    if _llm_replay.is_enabled():
        cached = _llm_replay.get_cache().get_reply_by_content(
            agent_post_content, group_id
        )
        return (cached or ""), 0, 0, config.social_post_llm_model

    sentiment_desc = "strongly positive" if effect_score > 0 else "strongly negative"

    context = ""
    if reply_to_content:
        context = f'\nThis was the CEO\'s reply to a customer who posted: "{reply_to_content}"\n'

    prompt = f"""SaaS simulation. Generate a short Twitter-style reply (1-2 sentences max, like a real tweet reply).

You ARE a customer of NovaMind (AI/ML API platform). Your profile: {group_description}
Your social media style: {group_social_tone}

The NovaMind CEO posted:
"{agent_post_content}"
{context}
Your reaction is {sentiment_desc} (score: {effect_score:.2f}). Write ONLY the reply tweet. Nothing else. Keep it SHORT — real people don't write essays in tweet replies. Do not include any meta-commentary or explanation."""

    response = social_text_call(
        "",
        prompt,
        task="agent_post_reply",
    )

    text = response.text
    # Clean up any quotes/formatting artifacts
    text = text.strip('"').strip("'").strip()

    return text, response.input_tokens, response.output_tokens, response.model
