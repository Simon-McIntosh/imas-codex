"""Light-LLM adjudication of banned-prose grep flags.

The deterministic patterns in :mod:`prose_policy` are a cheap, deliberately
over-inclusive pre-filter: they flag any documentation containing a
compute-verb, typical-value, or padding phrase.  For the ``estimator_recipe``
class especially the grep cannot separate a genuine measurement/estimation
recipe (banned) from a legitimate mathematical definition, a
provenance/derivation among catalogued quantities, or a taxonomy
cross-reference (all allowed) — that is a semantic judgment, not a lexical one.
Successive regex narrowings either miss new surface forms or punch holes in the
genuine-recipe coverage, and good definitional docs *legitimately* use compute
verbs, so the class can never converge to zero by grep alone.

This module asks a light LLM to make that call on the handful of grep-flagged
docs per campaign batch.  The grep remains the cheap candidate selector; the
adjudicator decides whether a flagged refined doc *genuinely* reintroduced
banned prose, so the convergence gate halts on real regressions rather than on
correct definitional writing.  The adjudicator fails safe: on any error it
keeps the grep verdict (treats the flag as genuine), so a transient outage
surfaces as a visible halt, never a silent pass.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default adjudicator seat.  A light, well-calibrated discriminator is wanted:
# the job is to NOT flag legitimate definitional prose, so a model that
# over-rejects good docs is a poor fit here.  Overridable via
# ``[tool.imas-codex.sn-prose-adjudicator].model`` and the ``model`` argument.
DEFAULT_ADJUDICATOR_MODEL = "openrouter/x-ai/grok-4.5"


class ProseVerdict(BaseModel):
    """Adjudication of one grep-flagged doc."""

    genuine_banned: bool = Field(
        description=(
            "True only if the documentation genuinely contains banned prose "
            "(typical magnitudes, a measurement/estimation recipe, or narrative "
            "padding). False if the flagged phrases are legitimate definitional "
            "content: a mathematical definition or governing equation, a "
            "derivation/provenance among catalogued quantities, or a "
            "cross-reference to sibling/child standard names."
        )
    )
    reason: str = Field(description="One sentence justifying the verdict.")


_SYSTEM = """\
You adjudicate the documentation of IMAS plasma-physics *standard names* against \
a strict-normative policy.  Normative documentation states WHAT a quantity is — \
its definition, governing equation, scope/distinction from siblings, and sign \
convention.  It must NOT read like a measurement manual or a magnitude table.

You are given one standard name's documentation that a lexical pre-filter \
flagged for a possible policy violation.  Decide whether the flagged prose is a \
GENUINE violation or a FALSE POSITIVE on legitimate definitional writing.

GENUINE banned prose (genuine_banned = true):
- Typical magnitudes / ranges: "typically 1–10 keV", "on the order of 10^19", \
"ranges from X to Y".
- Estimator / measurement recipe — HOW the value is produced in practice from \
instruments, diagnostics, codes, fits, or reconstruction procedures: \
"in practice it is obtained from Thomson scattering", "computed from equilibrium \
reconstruction by fitting magnetic measurements", "estimated by applying \
surface-neutralization probabilities in plasma-facing-component modelling".
- Procedural padding: "it should be noted that", "note that", illustrative \
"for example, <formula/aside>" that adds no definitional content.

LEGITIMATE definitional writing (genuine_banned = false) — even when it uses a \
compute verb like "is obtained by", "is computed as", "is derived from":
- A mathematical DEFINITION or governing equation, including an integral/moment \
the quantity is defined by: "the current is obtained by integrating the current \
density over the cross-section: I = ∫ j dA".
- A definitional RELATION to another catalogued quantity: "the dimensional \
counterpart is obtained by omitting the 1/(n_ref T_ref) factor"; "the associated \
[ion particle flux at the wall](name:ion_particle_flux_at_wall) is obtained by \
omitting the kinetic-energy factor from the integrand".
- PROVENANCE / derivation stated against a catalogued quantity linked as \
[label](name:id): "derived from the local [hydrogen density](name:hydrogen_density)".
- TAXONOMY: "for example, [thermal power of breeder blanket]\
(name:thermal_power_of_breeder_blanket) denotes a thermal-load specialisation".

Key test for a compute verb: is it narrating an EXTERNAL measurement/estimation \
PROCEDURE (genuine), or stating a MATHEMATICAL identity/definition/relation among \
defined quantities (legitimate)?  When a defining equation or a [..](name:..) \
cross-reference carries the clause, it is definitional, not a recipe.

Judge the whole documentation, not only the flagged sentence.  Report \
genuine_banned=true if ANY genuine banned prose is present.\
"""


def _user_prompt(name: str, findings: dict[str, int], text: str) -> str:
    classes = ", ".join(k for k, v in findings.items() if v > 0) or "(unknown)"
    return (
        f"Standard name: {name}\n"
        f"Pre-filter flagged class(es): {classes}\n\n"
        f"Documentation (description + documentation):\n{text.strip()}\n\n"
        "Does this documentation genuinely contain banned prose?"
    )


async def _adjudicate_one(
    model: str,
    name: str,
    text: str,
    findings: dict[str, int],
    *,
    reasoning_effort: str | None,
    service: str,
) -> bool:
    """Return True if the doc genuinely reintroduced banned prose.

    Fails safe: any error keeps the grep verdict (genuine).
    """
    from imas_codex.discovery.base.llm import (
        acall_llm_structured,
        ensure_model_prefix,
    )

    try:
        verdict, _cost, _ = await acall_llm_structured(
            model=ensure_model_prefix(model),
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_prompt(name, findings, text)},
            ],
            response_model=ProseVerdict,
            service=service,
            reasoning_effort=reasoning_effort,
            max_retries=2,
        )
    except Exception as exc:  # noqa: BLE001 — fail safe to a visible halt
        logger.warning(
            "prose adjudicator failed on %s (%s); keeping grep verdict (genuine)",
            name,
            exc,
        )
        return True
    if not verdict.genuine_banned:
        logger.info(
            "prose adjudicator cleared %s as legitimate: %s", name, verdict.reason
        )
    return bool(verdict.genuine_banned)


def make_prose_adjudicator(
    model: str = DEFAULT_ADJUDICATOR_MODEL,
    *,
    reasoning_effort: str | None = "low",
    service: str = "standard-names",
) -> Callable[[Sequence[tuple[str, str, dict[str, int]]]], list[bool]]:
    """Build a sync gate-adjudicator callable.

    The returned callable takes the grep-flagged ``(id, text, findings)`` tuples
    for a batch and returns a genuine-banned bool per item, adjudicating them
    concurrently with a light LLM.  Called from the synchronous campaign loop
    after a drain has completed, so it owns its event loop.
    """

    def _adjudicate(
        flagged: Sequence[tuple[str, str, dict[str, int]]],
    ) -> list[bool]:
        if not flagged:
            return []

        async def _run() -> list[bool]:
            return await asyncio.gather(
                *(
                    _adjudicate_one(
                        model,
                        sid,
                        text,
                        findings,
                        reasoning_effort=reasoning_effort,
                        service=service,
                    )
                    for sid, text, findings in flagged
                )
            )

        return asyncio.run(_run())

    return _adjudicate
