"""
Standalone health check for the Nexus LLM provider chain (nexus_llm).

Sends one small sample prompt through nexus_llm.call_chat — the exact path the
pipeline uses — so a green run here means the pipeline's LLM step will work.
Because call_chat tries providers in priority order (NVIDIA, then Azure) and
falls through on empty/failed responses, PASS means "at least one provider is
healthy right now"; the reply is prefixed with which provider answered is not
exposed by call_chat, so this reports overall reachability.

It also probes each configured provider individually so you can see, per
provider, whether it's healthy, empty-bodied (the known Azure/grok failure),
rate-limited, or misconfigured.

Usage:
    python test_llm.py
    python test_llm.py --no-chain     # only probe providers individually
"""

import argparse

from dotenv import load_dotenv

import nexus_llm

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Respond with a single JSON object and nothing else."},
    {"role": "user", "content": (
        'Extract the fields from this sentence and return them as JSON with keys '
        '"regulator", "document_type", "year": '
        '"The Reserve Bank of India issued a Master Direction on Credit Derivatives in 2026."'
    )},
]


def probe_each_provider() -> None:
    import time
    import httpx

    providers = nexus_llm._providers()
    if not providers:
        print("No providers configured. Set NVIDIA_* or AZURE_OPENAI_* in .env.")
        return

    print(f"Provider chain (priority order): {nexus_llm.configured_summary()}")
    print("-" * 60)
    for p in providers:
        body = {"model": p["model"], "messages": SAMPLE_MESSAGES}
        try:
            t0 = time.time()
            with httpx.Client(timeout=60) as client:
                resp = client.post(p["url"], headers=p["headers"], json=body)
            dt = round(time.time() - t0, 2)
            if resp.status_code != 200:
                verdict = f"HTTP {resp.status_code} — {resp.text[:120]}"
            elif not resp.text.strip():
                verdict = "EMPTY body (degraded — known Azure/grok failure mode)"
            else:
                try:
                    content = resp.json()["choices"][0]["message"]["content"]
                    verdict = f"OK — {content[:80].strip()}"
                except Exception as e:
                    verdict = f"200 but unparseable ({e})"
            print(f"  {p['name']:8s} ({p['model'][:32]:32s}) [{dt}s]  {verdict}")
        except Exception as e:
            print(f"  {p['name']:8s} ({p['model'][:32]:32s})  EXCEPTION {type(e).__name__}: {e}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Health-check the Nexus LLM provider chain.")
    parser.add_argument("--no-chain", action="store_true", help="Only probe providers individually; skip the call_chat test")
    args = parser.parse_args()

    load_dotenv(override=True)

    if not nexus_llm.is_configured():
        raise SystemExit(
            "No LLM provider configured — set NVIDIA_API_KEY/NVIDIA_BASE_URL/NVIDIA_MODEL "
            "or AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY in .env."
        )

    probe_each_provider()

    if args.no_chain:
        return

    print("call_chat (what the pipeline actually uses):")
    try:
        content = nexus_llm.call_chat(SAMPLE_MESSAGES)
    except nexus_llm.LLMError as e:
        print(f"  RESULT: FAIL — {e}")
        raise SystemExit(2)

    print(f"  model reply: {content.strip()[:200]}")
    print("\nRESULT: PASS — the provider chain returned real content; the pipeline's LLM step will work.")


if __name__ == "__main__":
    main()
