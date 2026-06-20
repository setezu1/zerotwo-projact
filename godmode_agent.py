"""
godmode_agent.py  —  G0DM0D3-inspired Multi-Model Intelligence Agent for Zero Two

Inspired by elder-plinius/G0DM0D3 (AGPL-3.0).
This implements the useful research features:
  - PARALLEL MODE: query Gemini + OpenAI + Anthropic simultaneously
  - AUTOTUNE: adaptive temperature/parameters based on task type
  - SCORER: rank responses by quality, return the winner
  - ULTRAMODE: for hard problems, run 3 variants per model with different params

What is NOT included: jailbreak prompts, safety bypasses, input perturbation.
Zero Two stays Zero Two — this just makes her smarter on hard questions.
"""

from __future__ import annotations

import threading
import time
import re
from typing import Optional
from pathlib import Path
import json
import sys


def _base_dir() -> Path:
    return Path(sys.executable).parent if getattr(sys,"frozen",False) else Path(__file__).resolve().parent

_CFG = _base_dir() / "config" / "api_keys.json"


def _load_keys() -> dict:
    try: return json.loads(_CFG.read_text(encoding="utf-8"))
    except Exception: return {}


# ── AutoTune: adaptive parameters ─────────────────────────────────────────────

def _autotune(prompt: str) -> dict:
    """
    Detect task type and return optimal generation parameters.
    Inspired by G0DM0D3's AutoTune module.
    """
    p = prompt.lower()
    # Creative / open-ended
    if any(w in p for w in ["write","story","poem","creative","imagine","invent","design"]):
        return {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1200}
    # Code / technical
    if any(w in p for w in ["code","function","fix","debug","python","script","implement","build"]):
        return {"temperature": 0.2, "top_p": 0.9, "max_tokens": 2000}
    # Analysis / reasoning
    if any(w in p for w in ["explain","analyze","why","how","compare","difference","research"]):
        return {"temperature": 0.4, "top_p": 0.9, "max_tokens": 1000}
    # Factual / lookup
    if any(w in p for w in ["what is","who is","when","where","define","list","facts"]):
        return {"temperature": 0.1, "top_p": 0.85, "max_tokens": 600}
    # Default
    return {"temperature": 0.7, "top_p": 0.92, "max_tokens": 800}


# ── Response scorer ────────────────────────────────────────────────────────────

def _score_response(prompt: str, response: str) -> float:
    """
    Simple quality scorer — returns 0.0 to 1.0.
    Checks: length appropriateness, avoids refusal phrases, specificity.
    """
    if not response or len(response.strip()) < 10:
        return 0.0

    score = 0.5

    # Penalize refusals
    refusal_phrases = [
        "i cannot","i can't","i'm unable","i am unable","i won't",
        "not able to","cannot assist","can't help","inappropriate",
        "against my","i don't have access"
    ]
    r_low = response.lower()
    if any(p in r_low for p in refusal_phrases):
        score -= 0.4

    # Reward length (appropriate to prompt length)
    prompt_len   = len(prompt.split())
    response_len = len(response.split())
    ideal_len    = max(50, prompt_len * 3)
    len_score    = min(1.0, response_len / ideal_len) * 0.3
    score += len_score

    # Reward specificity — numbers, code blocks, lists
    if re.search(r'\d+', response):      score += 0.05
    if '```' in response:                score += 0.1
    if '\n-' in response or '\n•' in response: score += 0.05

    # Penalize very short responses for complex prompts
    if prompt_len > 15 and response_len < 30:
        score -= 0.2

    return max(0.0, min(1.0, score))


# ── Provider callers ───────────────────────────────────────────────────────────

def _call_gemini(prompt: str, system: str, params: dict) -> str:
    import google.generativeai as genai
    key = _load_keys().get("gemini_api_key","")
    if not key: raise ValueError("No Gemini key")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system,
    )
    r = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=params.get("max_tokens",800),
            temperature=params.get("temperature",0.7),
        )
    )
    return r.text.strip()


def _call_openai(prompt: str, system: str, params: dict) -> str:
    try: import openai
    except ImportError: raise ImportError("pip install openai")
    key = _load_keys().get("openai_api_key","")
    if not key: raise ValueError("No OpenAI key")
    client = openai.OpenAI(api_key=key)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        max_tokens=params.get("max_tokens",800),
        temperature=params.get("temperature",0.7),
    )
    return r.choices[0].message.content.strip()


def _call_anthropic(prompt: str, system: str, params: dict) -> str:
    try: import anthropic
    except ImportError: raise ImportError("pip install anthropic")
    key = _load_keys().get("anthropic_api_key","")
    if not key: raise ValueError("No Anthropic key")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=params.get("max_tokens",800),
        system=system,
        messages=[{"role":"user","content":prompt}],
    )
    return msg.content[0].text.strip()


_CALLERS = {
    "gemini":    _call_gemini,
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
}


# ── GodMode Agent ──────────────────────────────────────────────────────────────

class GodModeAgent:
    """
    G0DM0D3-inspired multi-model parallel query agent.
    Queries available providers simultaneously, scores responses, returns winner.
    Falls back gracefully if only one provider is configured.
    """

    def __init__(self):
        self._last_params: dict = {}

    def available_providers(self) -> list[str]:
        keys = _load_keys()
        avail = []
        if keys.get("gemini_api_key"):    avail.append("gemini")
        if keys.get("openai_api_key"):    avail.append("openai")
        if keys.get("anthropic_api_key"): avail.append("anthropic")
        return avail

    def query(
        self,
        prompt:    str,
        system:    str  = "You are Zero Two, a smart and slightly playful AI assistant. Be concise and direct.",
        ultra:     bool = False,
        log_fn     = None,
    ) -> str:
        """
        Query all available providers in parallel, score, return best answer.
        ultra=True: runs 2 temperature variants per provider (slower but better).
        """
        providers = self.available_providers()
        if not providers:
            return ""

        params = _autotune(prompt)
        self._last_params = params

        if log_fn:
            mode = "ULTRA" if ultra else "PARALLEL"
            log_fn(f"SYS: GodMode {mode} — querying {len(providers)} provider(s)")

        results: dict[str, str]   = {}
        errors:  dict[str, str]   = {}
        lock = threading.Lock()

        def _run(provider: str, p: dict):
            try:
                fn   = _CALLERS[provider]
                resp = fn(prompt, system, p)
                with lock:
                    results[f"{provider}"] = resp
            except Exception as e:
                with lock:
                    errors[provider] = str(e)
                    print(f"[GodMode] {provider} failed: {e}")

        threads = []
        variants = [params]
        if ultra and len(providers) >= 2:
            # Two temperature variants for higher-quality parallel race
            hi = dict(params, temperature=min(1.0, params["temperature"] + 0.3))
            variants = [params, hi]

        for provider in providers:
            for i, p in enumerate(variants):
                key = f"{provider}" if len(variants)==1 else f"{provider}_v{i}"
                t = threading.Thread(target=_run, args=(provider, p), daemon=True)
                threads.append(t)
                t.start()

        # Wait for all (timeout 20s)
        for t in threads:
            t.join(timeout=20)

        if not results:
            if errors:
                return f"All providers failed: {'; '.join(f'{k}: {v}' for k,v in errors.items())}"
            return ""

        # Score and pick winner
        scored = {k: (_score_response(prompt, v), v) for k, v in results.items()}
        winner_key = max(scored, key=lambda k: scored[k][0])
        best_score, best_resp = scored[winner_key]

        if log_fn:
            scores_str = " | ".join(f"{k}={s:.2f}" for k,(s,_) in scored.items())
            log_fn(f"SYS: GodMode winner: {winner_key} (score={best_score:.2f}) [{scores_str}]")

        return best_resp

    def status(self) -> dict:
        return {
            "providers": self.available_providers(),
            "last_params": self._last_params,
        }


# Singleton
_godmode = GodModeAgent()

def get_godmode() -> GodModeAgent:
    return _godmode
