"""AI Sharks: five investors, two rounds, and a host summary.

Python 3.12+; install langchain-groq, langgraph, pydantic, python-dotenv.
Run: python ai_sharks_structured.py [optional_pitch.txt]
Notebook: import this module, then await run_panel(pitch).
The original dependency pins are unverified; use a compatible environment.
API calls incur provider usage. Default run: 10 verdict calls + 1 summary.
The original supervisor experiment is replaced with deterministic routing.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from contextvars import ContextVar
from getpass import getpass
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# 1. Investor personas
@dataclass
class Shark:
    name: str
    title: str
    system: str


SHARKS = [
    Shark(
        "Nikhil", "The Numbers Shark",
        "You are Nikhil, a ruthless numbers-first investor. You judge a startup on unit "
        "economics alone: margins, CAC vs LTV, burn rate, path to profit, and whether the "
        "valuation is sane. You are unmoved by vision or hype - show you the math or you pass. "
        "You speak bluntly and quote numbers.",
    ),
    Shark(
        "Veena", "The Visionary",
        "You are Veena, a visionary investor. You back bold founders and enormous markets. You "
        "look past today's messy numbers to the 10x outcome - the mission, the timing, the "
        "founder's ambition. You get genuinely excited by a big idea and are willing to bet early. "
        "You speak with warmth and conviction.",
    ),
    Shark(
        "Sameer", "The Skeptic",
        "You are Sameer, a sharp skeptic. Your job is to find why this startup will FAIL: crowded "
        "competition, no moat, weak differentiation, regulatory or execution risk, a founder gap. "
        "You are hard to convince and you probe the weakest point. You are fair but tough, and you "
        "say plainly what worries you.",
    ),
]

HOST = (
    "You are the host of 'Pitch to the AI Sharks'. You are witty, fair and concise. In 3-4 "
    "sentences, summarise how the panel voted, note where the sharks agreed and where they "
    "clashed, and leave the founder with one clear, actionable takeaway."
)


priya = Shark(
    "Priya", "The Customer Shark",
    "You are Priya, a customer-first investor. You focus deeply on customer pain points, "
    "willingness to pay, retention, and genuine product-market fit. You demand clear evidence "
    "that customers actually want and love this product. If a pitch assumes customers will buy "
    "without validating demand, you reject it. You speak with empathy but hold a high bar for user feedback."
)

arjun = Shark(
    "Arjun", "The Growth Shark",
    "You are Arjun, an ambitious, data-driven, and commercially minded growth investor. You "
    "focus on customer acquisition channels, distribution strategy, scalability, and market "
    "expansion potential. You ruthlessly question unrealistic growth projections and "
    "unsustainable acquisition costs (CAC). You speak dynamically and demand scale."
)


SHARKS.extend([priya, arjun])

# 2. Validated response and graph state
class Verdict(BaseModel):
    shark: str = Field(description="Investor name")
    stance: Literal["INVEST", "PASS"]
    offer: str | None = Field(description="Investment offer, or null for PASS")
    reasoning: str = Field(description="Two or three sentences in the investor's voice")


class PanelState(TypedDict):
    pitch: str
    round1: list[Verdict]
    final: list[Verdict]
    summary: str


# 3. Configuration and reusable helpers
_SESSION_KEY: ContextVar[str | None] = ContextVar("groq_session_key", default=None)
MODEL = "openai/gpt-oss-120b"
MAX_CONCURRENT_CALLS = 2


def configure() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        key = getpass("Groq API key (hidden): ").strip()
    if not key:
        raise ValueError("GROQ_API_KEY is required.")
    os.environ["GROQ_API_KEY"] = key


def new_llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        model=MODEL if _SESSION_KEY.get() else os.getenv("GROQ_MODEL", MODEL),
        api_key=_SESSION_KEY.get() or os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        max_retries=2,
        timeout=60,
    )


def format_verdicts(verdicts: list[Verdict]) -> str:
    return "\n".join(
        f"- {v.shark} — {v.stance}: {v.reasoning}"
        + (f" Offer: {v.offer}" if v.offer else "")
        for v in verdicts
    )


async def evaluate_async(shark: Shark, pitch: str, others: str = "") -> Verdict:
    prompt = f"Judge this startup pitch:\n{pitch}"
    if others:
        prompt += (
            f"\nOther investors' first-round verdicts:\n{others}"
            "\nReact where you disagree, then make your final decision."
        )
    prompt += (
        f"\nRespond as {shark.name}. Decide INVEST or PASS. "
        "Provide an offer only for INVEST; use null for PASS."
    )
    # Requires a Groq model that supports strict JSON schema output.
    verdict = await new_llm().with_structured_output(
        Verdict, method="json_schema", strict=True
    ).ainvoke([("system", shark.system), ("human", prompt)])
    verdict.shark = shark.name
    if verdict.stance == "PASS":
        verdict.offer = None
    elif not verdict.offer or not verdict.offer.strip():
        raise ValueError(f"{shark.name} returned INVEST without an offer.")
    return verdict


# 4. Workflow: independent round, peer reaction, host summary
async def evaluate_round(pitch: str, previous: list[Verdict] | None = None) -> list[Verdict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    async def evaluate(shark: Shark) -> Verdict:
        others = format_verdicts([
            v for v in (previous or []) if v.shark != shark.name
        ])
        async with semaphore:
            return await evaluate_async(shark, pitch, others)

    tasks = [asyncio.create_task(evaluate(shark)) for shark in SHARKS]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def panel_node(state: PanelState) -> dict:
    return {"round1": await evaluate_round(state["pitch"])}


async def debate_node(state: PanelState) -> dict:
    return {"final": await evaluate_round(state["pitch"], state["round1"])}


async def host_node(state: PanelState) -> dict:
    prompt = (
        f"Pitch:\n{state['pitch']}\n\nFinal verdicts:\n"
        + format_verdicts(state["final"])
        + "\nGive your wrap-up."
    )
    response = await new_llm(0.5).ainvoke([
        ("system", HOST), ("human", prompt)
    ])
    return {"summary": response.content}


def build_panel():
    graph = StateGraph(PanelState)
    graph.add_node("panel", panel_node)
    graph.add_node("debate", debate_node)
    graph.add_node("host", host_node)
    graph.set_entry_point("panel")
    graph.add_edge("panel", "debate")
    graph.add_edge("debate", "host")
    graph.add_edge("host", END)
    return graph.compile()


async def run_panel(pitch: str, *, api_key: str | None = None) -> PanelState:
    if not pitch.strip():
        raise ValueError("Pitch cannot be empty.")
    if api_key is None:
        configure()
    elif not api_key.strip():
        raise ValueError("Groq API key is required.")
    token = _SESSION_KEY.set(api_key.strip() if api_key is not None else None)
    try:
        result = await build_panel().ainvoke({
            "pitch": pitch, "round1": [], "final": [], "summary": ""
        })
        if any(len(result[key]) != len(SHARKS) for key in ("round1", "final")):
            raise RuntimeError("Incomplete panel results.")
        return result
    finally:
        _SESSION_KEY.reset(token)


# 5. Sample pitch and command-line entry point
pitch = 'GharSolar - rooftop solar subscriptions for middle-class Indian homes.\n\nProblem: rooftop solar saves families 60% on power bills, but the Rs 2-3 lakh upfront cost\nputs it out of reach for most.\n\nSolution: GharSolar installs solar for zero upfront cost; households pay a fixed monthly\nsubscription lower than their old bill. We own the panels and the financing.\n\nTraction: 2,100 homes installed across Jaipur and Indore in 14 months. Rs 1.1 crore monthly\nrecurring revenue. BUT gross margin is only 12% today because of hardware and install costs,\nand each install costs us Rs 1.8 lakh to acquire, recovered over 4 years. Churn is under 2%.\n\nAsk: raising Rs 25 crore for 15% to build install capacity and negotiate panel supply deals.\n'

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pitch_file", nargs="?", type=Path)
    args = parser.parse_args()
    selected_pitch = args.pitch_file.read_text(encoding="utf-8") if args.pitch_file else pitch
    result = await run_panel(selected_pitch)
    print("\nFIRST ROUND\n" + format_verdicts(result["round1"]))
    print("\nFINAL VERDICTS\n" + format_verdicts(result["final"]))
    print("\nHOST SUMMARY\n" + result["summary"])


if __name__ == "__main__":
    asyncio.run(main())
