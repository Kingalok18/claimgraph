"""Short plain labels for topics, from the small LLM. Falls back to the claims' own words."""
from __future__ import annotations

import logging
import re
from collections import Counter

from langchain_core.output_parsers import StrOutputParser

from ..claims.extract_chain import load_prompt
from ..llm import get_chat_model

log = logging.getLogger(__name__)
PROMPT_VERSION = "label_topic.v1"
_BANNED = re.compile(r"\b(ai-powered|powerful|magic|smart|novel|insights?|unlock|revolutionary|cutting[- ]edge)\b", re.I)


def build_label_chain(llm=None):
    model = llm if llm is not None else get_chat_model("small")
    chain = load_prompt(PROMPT_VERSION) | model | StrOutputParser()
    return chain.with_config(metadata={"chain_name": "label_topic", "prompt_version": PROMPT_VERSION})


def clean_label(raw: str) -> str:
    line = next((item for item in raw.splitlines() if item.strip()), "")
    line = re.sub(r"^(label|topic)\s*:\s*", "", line.strip(), flags=re.I).strip(" \"'.*#`")
    line = _BANNED.sub("", line)
    line = re.sub(r"\s{2,}", " ", line).strip(" ,;:-")
    return " ".join(line.split()[:8])


def fallback_label(claims: list[dict]) -> str:
    first = Counter(claim["intervention"].strip() for claim in claims).most_common(1)[0][0]
    second = Counter(claim["outcome"].strip() for claim in claims).most_common(1)[0][0]
    return " ".join(f"{first}: {second}".split()[:8]) or "Unnamed topic"


def label_topic(claims: list[dict], chain=None) -> str:
    """Label from up to 8 claims. Any failure gives the fallback; a label is never worth stopping for."""
    try:
        chain = chain if chain is not None else build_label_chain()
        listing = "\n".join(f"- {claim['text']}" for claim in claims[:8])
        label = clean_label(chain.invoke({"claims": listing}))
        if len(label) >= 3:
            return label
    except Exception as exc:  # noqa: BLE001
        log.warning("Topic label failed, using the fallback: %s", exc)
    return fallback_label(claims)
