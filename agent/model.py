"""Model wrappers. The loop only talks to `respond()`, so tests can use a fake model."""
import json
import os
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def content_blocks(self) -> list[dict]:
        """The assistant message to put back in the history."""
        blocks = [{"type": "text", "text": self.text}] if self.text else []
        blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in self.tool_calls]
        return blocks


class AnthropicModel:
    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import anthropic  # imported here so tests run without an API key

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def respond(self, system: str, messages: list, tools: list) -> ModelTurn:
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, tools=tools
        )
        turn = ModelTurn(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)
        for b in r.content:
            if b.type == "text":
                turn.text += b.text
            elif b.type == "tool_use":
                turn.tool_calls.append(ToolCall(b.id, b.name, dict(b.input)))
        return turn


class OpenAIModel:
    """Same `respond()` contract as AnthropicModel, backed by OpenAI's chat.completions API.

    The loop and toolbox speak Anthropic-shaped messages/tool schemas (tool_use / tool_result
    blocks, `input_schema`), so this class just translates to and from OpenAI's shapes on every
    call. Nothing in loop.py or tools.py needs to know which provider is in use.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import openai  # imported here so tests run without an API key

        self.client = openai.OpenAI()  # reads OPENAI_API_KEY
        # gpt-4o-mini is a plain (non-reasoning) model: every output token is billed and
        # visible, unlike gpt-5-mini/nano where hidden "reasoning" tokens eat the budget
        # before any text is written. For a tight budget this is both cheaper per call and
        # far more predictable (no risk of a turn truncating to empty text).
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.max_tokens = max_tokens

    @staticmethod
    def _to_openai_messages(system: str, messages: list) -> list:
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            if m["role"] == "assistant":
                text = "".join(b["text"] for b in content if b["type"] == "text")
                tool_calls = [
                    {"id": b["id"], "type": "function",
                     "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                    for b in content if b["type"] == "tool_use"
                ]
                msg = {"role": "assistant", "content": text or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            else:  # a "user" message carrying tool_result blocks
                for b in content:
                    if b["type"] == "tool_result":
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
        return out

    @staticmethod
    def _to_openai_tools(tools: list) -> list:
        return [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ]

    def respond(self, system: str, messages: list, tools: list) -> ModelTurn:
        r = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=self._to_openai_messages(system, messages),
            tools=self._to_openai_tools(tools),
        )
        msg = r.choices[0].message
        usage = r.usage
        turn = ModelTurn(
            text=msg.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        for tc in msg.tool_calls or []:
            turn.tool_calls.append(ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments)))
        return turn


class ScriptedModel:
    """Plays back a fixed list of turns. Each turn is a list of (name, input) or a final string."""

    def __init__(self, script: list, tokens_per_turn: int = 100):
        self.script = list(script)
        self.tokens_per_turn = tokens_per_turn
        self.n = 0

    def respond(self, system, messages, tools) -> ModelTurn:
        if not self.script:
            return ModelTurn(text="Done.")
        step = self.script.pop(0)
        if isinstance(step, str):
            return ModelTurn(text=step, input_tokens=self.tokens_per_turn)
        calls = []
        for name, args in step:
            self.n += 1
            calls.append(ToolCall(f"call_{self.n}", name, args))
        return ModelTurn(tool_calls=calls, input_tokens=self.tokens_per_turn)


class LoopingModel:
    """A broken model that never stops calling tools. Used to prove the stop rule holds."""

    def __init__(self):
        self.n = 0

    def respond(self, system, messages, tools) -> ModelTurn:
        self.n += 1
        return ModelTurn(tool_calls=[ToolCall(f"loop_{self.n}", "list_invoices", {})], input_tokens=50)
