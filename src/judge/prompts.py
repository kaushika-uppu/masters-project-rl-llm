"""Prompts for the step-validity judge.

The judge evaluates step soundness INTRINSICALLY (path-agnostic) — it is NOT shown the
reference proof, so it cannot suppress valid alternative paths. It also maintains the
running proof state (established results + current goal) used as the node-merge key.
"""

JUDGE_SYSTEM_PROMPT = (
    "You are a rigorous mathematics proof checker. You are given a claim, the proof "
    "state so far (established results and current goal), and ONE candidate next step. "
    "Decide ONLY whether that step is a logically sound inference from the current "
    "state. Judge soundness on its own merits — there are many valid ways to prove a "
    "claim, so do NOT require any particular approach. Then update the proof state.\n\n"
    "Respond with STRICT JSON and nothing else:\n"
    '{"valid": true|false, "reason": "<one sentence>", '
    '"state_summary": "<updated established results + current goal>", '
    '"is_terminal": true|false, "verdict": "PROVED"|"DISPROVED"|null}\n'
    "Set is_terminal=true only if this step completes the argument; then set verdict "
    "to PROVED if the claim is established true, or DISPROVED if shown false."
)


def build_judge_user_prompt(statement: str, history: list[str], step: str) -> str:
    state = "\n".join(f"- {h}" for h in history) if history else "(no steps yet; goal is the claim)"
    return (
        f"CLAIM (prove or disprove):\n{statement}\n\n"
        f"PROOF STATE SO FAR:\n{state}\n\n"
        f"CANDIDATE NEXT STEP:\n{step}\n\n"
        "Return the JSON verdict for this step."
    )
