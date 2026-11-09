import anthropic

SYSTEM_PROMPT = """\
You answer questions about the user's documents. You get numbered excerpts from \
those documents, retrieved by a search system.

- Base your answer only on the excerpts. If they don't contain the answer, say so \
plainly instead of guessing, and mention what the excerpts do cover.
- Cite every statement with the excerpt number in square brackets, e.g. [1] or [2][4], \
directly after the sentence it supports.
- Answer in the language of the question. Be concise; use short lists where they help."""


class AnswerError(Exception):
    pass


def format_sources(sources):
    parts = []
    for n, s in enumerate(sources, start=1):
        c = s.chunk
        parts.append(f'<excerpt n="{n}" file="{c.filename}" page="{c.page}">\n{c.text}\n</excerpt>')
    return "\n\n".join(parts)


def build_messages(question, sources, history):
    messages = []
    for m in history[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    # api will dass die erste nachricht vom user kommt
    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    excerpts = format_sources(sources) if sources else "(no matching excerpts found)"
    messages.append(
        {"role": "user", "content": f"<excerpts>\n{excerpts}\n</excerpts>\n\nQuestion: {question}"}
    )
    return messages


class ClaudeGenerator:
    def __init__(self, model, effort, max_tokens):
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def stream(self, question, sources, history):
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=build_messages(question, sources, history),
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise AnswerError("API-Key ungültig, check mal die .env") from e
        except anthropic.RateLimitError as e:
            raise AnswerError("Rate-Limit erreicht, kurz warten") from e
        except anthropic.APIConnectionError as e:
            raise AnswerError("Keine Verbindung zur Claude API") from e
        except anthropic.APIStatusError as e:
            raise AnswerError(f"Fehler von der Claude API ({e.status_code})") from e

        if final.stop_reason == "refusal":
            raise AnswerError("Claude hat die Anfrage abgelehnt")
        if final.stop_reason == "max_tokens":
            yield "\n\n(Antwort abgeschnitten)"
