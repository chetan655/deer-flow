"""Summarization middleware extensions for DeerFlow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain.agents import AgentState
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummarizationEvent:
    """Context emitted before conversation history is summarized away."""

    messages_to_summarize: tuple[AnyMessage, ...]
    preserved_messages: tuple[AnyMessage, ...]
    thread_id: str | None
    agent_name: str | None
    runtime: Runtime


@runtime_checkable
class BeforeSummarizationHook(Protocol):
    """Hook invoked before summarization removes messages from state."""

    def __call__(self, event: SummarizationEvent) -> None: ...


def _resolve_thread_id(runtime: Runtime) -> str | None:
    """Resolve the current thread ID from runtime context or LangGraph config."""
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id is None:
        try:
            config_data = get_config()
        except RuntimeError:
            return None
        thread_id = config_data.get("configurable", {}).get("thread_id")
    return thread_id


def _resolve_agent_name(runtime: Runtime) -> str | None:
    """Resolve the current agent name from runtime context or LangGraph config."""
    agent_name = runtime.context.get("agent_name") if runtime.context else None
    if agent_name is None:
        try:
            config_data = get_config()
        except RuntimeError:
            return None
        agent_name = config_data.get("configurable", {}).get("agent_name")
    return agent_name


class DeerFlowSummarizationMiddleware(SummarizationMiddleware):
    """Summarization middleware with pre-compression hook dispatch."""

    def __init__(
        self,
        *args,
        before_summarization: list[BeforeSummarizationHook] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._before_summarization_hooks = before_summarization or []

    def _build_new_messages(self, summary: str) -> list[AnyMessage]:
        """Override base behaviour to strictly cast the summary as a SystemMessage."""
        return [SystemMessage(content=f"Here is a summary of the conversation to date:\n{summary}")]

    def _rebuild_messages_with_anchor(self, original_messages: list[AnyMessage], new_messages: list[AnyMessage], preserved_messages: list[AnyMessage]) -> dict:
        current_msg = next((msg for msg in reversed(original_messages) if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human"), None)

        current_msg_id = getattr(current_msg, "id", None) if current_msg else None

        is_preserved = False
        if current_msg_id:
            is_preserved = any(getattr(msg, "id", None) == current_msg_id for msg in preserved_messages)

        final_messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        final_messages.extend(new_messages)

        if current_msg and not is_preserved:
            final_messages.append(current_msg)

        final_messages.extend(preserved_messages)
        return {"messages": final_messages}

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._maybe_summarize(state, runtime)

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return await self._amaybe_summarize(state, runtime)

    def _maybe_summarize(self, state: AgentState, runtime: Runtime) -> dict | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)
        self._fire_hooks(messages_to_summarize, preserved_messages, runtime)

        summary_text = self._create_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary_text)

        return self._rebuild_messages_with_anchor(messages, new_messages, preserved_messages)

    async def _amaybe_summarize(self, state: AgentState, runtime: Runtime) -> dict | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)
        self._fire_hooks(messages_to_summarize, preserved_messages, runtime)

        summary_text = await self._acreate_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary_text)

        return self._rebuild_messages_with_anchor(messages, new_messages, preserved_messages)

    def _fire_hooks(
        self,
        messages_to_summarize: list[AnyMessage],
        preserved_messages: list[AnyMessage],
        runtime: Runtime,
    ) -> None:
        if not self._before_summarization_hooks:
            return

        event = SummarizationEvent(
            messages_to_summarize=tuple(messages_to_summarize),
            preserved_messages=tuple(preserved_messages),
            thread_id=_resolve_thread_id(runtime),
            agent_name=_resolve_agent_name(runtime),
            runtime=runtime,
        )

        for hook in self._before_summarization_hooks:
            try:
                hook(event)
            except Exception:
                hook_name = getattr(hook, "__name__", None) or type(hook).__name__
                logger.exception("before_summarization hook %s failed", hook_name)
