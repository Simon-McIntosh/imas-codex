"""Store a standard name's Jev domain decision and write its domain through one writer.

A standard name's physics domain is not authored. Its ``source_domains`` is
exactly the set of its current DD sources' domains, and a derived parent's is
the union of its children's. Its ``physics_domain`` is the measured Jev Choice
over the candidates the Data Dictionary supplies for it: the stored domains of
its source paths, its source IDSs' mapped domains, the band umbrella of each
band those candidates fall in, and ``general``.

One function, :func:`write_name_domain`, sets ``physics_domain``,
``source_domains`` and the ``HAS_PHYSICS_DOMAIN`` edges together in one
transaction, so the three cannot disagree. Every writer of a name's domain
routes through it.

The decision is stored on the name — the candidates offered, the choice, its
probability, the model and when it was decided — so it is reproducible and the
audit can check it. A recorded correction sits beside it: the domain and its
reason. When a name does not clear :data:`CONFIDENCE_FLOOR`, or its call fails,
the name is *undecided*: it keeps its current domain and is reported with the
reason. It never receives a guessed one.

The bands, umbrellas and scope text come from
``imas_codex/definitions/physics/domains.yaml`` — the one source of the
guidance — read through the loader here, so a name's decision uses the same
guidance the measurement did.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import yaml

from imas_codex.definitions.physics import DOMAINS_SCHEMA, IDS_DOMAINS_FILE
from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.dd_domain_classifier import SERVICE_TAG

__all__ = [
    "CONFIDENCE_FLOOR",
    "CONCURRENCY",
    "QUESTION",
    "SEAT",
    "DomainCorrection",
    "DomainDecision",
    "NameDomainWrite",
    "band_of",
    "band_umbrella",
    "build_question",
    "build_state",
    "candidate_domains",
    "decide_domain",
    "ids_domain_map",
    "option_text",
    "write_name_domain",
]

SEAT = "sn-domain-tree"
QUESTION = "physics_domain"
CONCURRENCY = 8

#: The confidence floor a name's chosen probability must clear to be decided.
#: Derived from the calibration names' measured answers as the highest observed
#: probability at which at most 5% of the calibration names fall strictly below
#: it; the derivation and the accuracy above it are recorded in
#: ``docs/evidence/fragments/sn-catalog-grouping/scg-name-domain-writer.html``.
CONFIDENCE_FLOOR: float = 0.68


def _pretty(raw: str) -> str:
    return " ".join(raw.split())


@lru_cache(maxsize=1)
def _guidance() -> dict[str, Any]:
    data = yaml.safe_load(DOMAINS_SCHEMA.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _bands() -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    """The band names, the domain-to-band map and the band-to-umbrella map."""
    bands = tuple(_guidance().get("precedence") or ())
    band_of_domain = {d: band["band"] for band in bands for d in band["domains"]}
    umbrella = {band["band"]: band["umbrella"] for band in bands}
    return tuple(band["band"] for band in bands), band_of_domain, umbrella


def band_of(domain: str) -> str | None:
    return _bands()[1].get(domain)


def band_umbrella(domain: str) -> str | None:
    band = band_of(domain)
    return _bands()[2].get(band) if band else None


def option_text(domain: str) -> str:
    """One domain's scope, includes and excludes from the guidance file."""
    meta = (_guidance().get("domains") or {}).get(domain) or {}
    parts = [str(meta.get("scope") or "").strip()]
    if meta.get("includes"):
        parts.append("Includes: " + "; ".join(str(i) for i in meta["includes"]))
    for exclusion in meta.get("excludes") or []:
        parts.append(
            "Excludes {quantity} (-> {neighbour})".format(
                quantity=exclusion.get("quantity", ""),
                neighbour=exclusion.get("neighbour", ""),
            )
            + (f": {exclusion['rule']}" if exclusion.get("rule") else "")
        )
    return "\n".join(part for part in parts if part)


def ids_domain_map() -> dict[str, str]:
    """Each IDS's mapped physics domain, from the version-independent map."""
    payload = json.loads(IDS_DOMAINS_FILE.read_text(encoding="utf-8"))
    return dict(payload["ids_domain_mappings"])


def ids_of(path: str) -> str:
    return path.removeprefix("dd:").split("/", 1)[0]


def candidate_domains(
    sources: Mapping[str, str | None], ids_map: Mapping[str, str]
) -> tuple[str, ...]:
    """The domains the DD supplies for a name: its paths', its IDSs' and umbrellas."""
    found: set[str] = {"general"}
    for path, domain in sources.items():
        if domain:
            found.add(domain)
        prior = ids_map.get(ids_of(path))
        if prior:
            found.add(prior)
    for domain in list(found):
        umbrella = band_umbrella(domain)
        if umbrella:
            found.add(umbrella)
    return tuple(sorted(found))


def path_counts(sources: Mapping[str, str | None]) -> dict[str, int]:
    counts = Counter(d for d in sources.values() if d)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_state(
    name: str,
    description: str,
    sources: Mapping[str, str | None],
    ids_map: Mapping[str, str],
    options: Sequence[str],
) -> dict[str, Any]:
    """The state a name's Choice is asked of, carrying its path-domain counts."""
    counts = path_counts(sources)
    return {
        "name": name,
        "description": description or "",
        "source_paths": [
            {"path": path, "stored_domain": domain or "unclassified"}
            for path, domain in sorted(sources.items())
        ],
        "source_ids": [
            {"ids": ids, "graph_domain": ids_map.get(ids) or "unclassified"}
            for ids in sorted({ids_of(path) for path in sources})
        ],
        "source_path_count_by_candidate_domain": {
            domain: counts.get(domain, 0) for domain in options
        },
    }


def build_question(options: Sequence[str]) -> dict[str, Any]:
    """The Choice, most specific option first, each carrying its scope text."""
    instructions = (
        "Choose the one physics domain that best describes the standard name "
        "given in the state, judging what the quantity measures rather than "
        "which IDS carries it. Every option is a domain the Data Dictionary "
        "supplies for this name: a stored domain of one of its source paths, "
        "the domain of one of its source IDSs, a broader band umbrella one of "
        "those falls under, or general. Prefer the most specific option that "
        "plainly fits the quantity; choose general only when no narrower "
        "option fits. The options are listed most specific first."
    )
    order = sorted(options, key=lambda d: (band_of(d) != "general", d))
    return {
        QUESTION: {
            "type": "choice",
            "instructions": instructions,
            "criteria": {domain: option_text(domain) for domain in order},
        }
    }


@dataclass(frozen=True)
class DomainDecision:
    """A name's measured domain decision, or the reason it is undecided."""

    name: str
    candidates: tuple[str, ...]
    choice: str | None
    probability: float | None
    model: str | None
    decided_at: str | None
    undecided_reason: str | None = None

    @property
    def decided(self) -> bool:
        return self.choice is not None and self.undecided_reason is None


@dataclass(frozen=True)
class DomainCorrection:
    """A lead-approved domain for a name the Choice gets wrong, with its reason."""

    domain: str
    reason: str


def _undecided(name: str, options: Sequence[str], reason: str) -> DomainDecision:
    return DomainDecision(
        name=name,
        candidates=tuple(options),
        choice=None,
        probability=None,
        model=None,
        decided_at=None,
        undecided_reason=reason,
    )


async def decide_domain(
    name: str,
    description: str,
    sources: Mapping[str, str | None],
    ids_map: Mapping[str, str],
    *,
    model: str,
) -> DomainDecision:
    """Ask Jev one name's domain Choice and return the decision it stored.

    A name with a single candidate takes it without a call. A call that fails,
    or whose choice does not clear :data:`CONFIDENCE_FLOOR`, is undecided and
    named with its reason; the same for a choice outside the candidates.
    """
    options = candidate_domains(sources, ids_map)
    if len(options) == 1:
        return DomainDecision(
            name=name,
            candidates=options,
            choice=options[0],
            probability=1.0,
            model=model,
            decided_at=datetime.now(UTC).isoformat(),
        )
    results, _cost = await decide_batch(
        [build_state(name, description, sources, ids_map, options)],
        build_question(options),
        model=model,
        service=SERVICE_TAG,
        concurrency=CONCURRENCY,
    )
    answer: dict[str, Any] | None = None
    if results and results[0]:
        answers = results[0][0]
        if isinstance(answers, dict):
            answer = answers.get(QUESTION)
    if not isinstance(answer, dict):
        return _undecided(name, options, "the Jev call returned no answer")
    choice = answer.get("choice")
    probabilities = {
        str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
    }
    probability = probabilities.get(str(choice))
    if choice not in options:
        return _undecided(
            name, options, f"the choice {choice!r} is not one of the candidates"
        )
    if probability is None:
        return _undecided(name, options, "the answer carried no probability")
    if probability < CONFIDENCE_FLOOR:
        return _undecided(
            name,
            options,
            f"chosen probability {probability:.3f} is below the floor "
            f"{CONFIDENCE_FLOOR:.2f}",
        )
    return DomainDecision(
        name=name,
        candidates=options,
        choice=str(choice),
        probability=probability,
        model=model,
        decided_at=datetime.now(UTC).isoformat(),
    )


def decide_domain_for(
    name: str,
    description: str,
    sources: Mapping[str, str | None],
    ids_map: Mapping[str, str],
    *,
    model: str,
) -> DomainDecision:
    """Synchronous face of :func:`decide_domain`."""
    return asyncio.run(
        decide_domain(name, description, sources, ids_map, model=model)
    )


@dataclass(frozen=True)
class NameDomainWrite:
    """What one writer call set on a name, and why."""

    sn_id: str
    physics_domain: str | None
    source_domains: tuple[str, ...]
    decided: bool
    correction: str | None
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sn_id": self.sn_id,
            "physics_domain": self.physics_domain,
            "source_domains": list(self.source_domains),
            "decided": self.decided,
            "correction": self.correction,
            "reason": self.reason,
        }


#: The name's current state in one read: its own DD source domains through
#: ``StandardName <-PRODUCED_NAME- StandardNameSource -FROM_DD_PATH-> IMASNode``
#: (empty for a derived parent), its children's DD domains and stored
#: ``source_domains``, its current scalar and the decision already stored.
_READ_STATE = """
MATCH (sn:StandardName {id: $sn_id})
CALL (sn) {
  OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
  OPTIONAL MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
  RETURN [d IN collect(DISTINCT dd.physics_domain) WHERE d IS NOT NULL] AS own_domains
}
CALL (sn) {
  OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
  OPTIONAL MATCH (child_src:StandardNameSource)-[:PRODUCED_NAME]->(child)
  OPTIONAL MATCH (child_src)-[:FROM_DD_PATH]->(child_dd:IMASNode)
  RETURN collect(DISTINCT child.id) AS child_ids,
         [d IN collect(DISTINCT child_dd.physics_domain) WHERE d IS NOT NULL]
           AS child_dd_domains,
         [c IN collect(DISTINCT child) WHERE c IS NOT NULL | c.source_domains]
           AS child_stored_domains
}
RETURN sn.physics_domain AS physics_domain,
       own_domains, child_ids, child_dd_domains, child_stored_domains
"""

#: One transaction for all three: the two scalars, the decision and correction
#: slots, and the edges rewritten to exactly ``$edge_domains``.
_WRITE_DOMAIN = """
MATCH (sn:StandardName {id: $sn_id})
OPTIONAL MATCH (sn)-[old:HAS_PHYSICS_DOMAIN]->(:PhysicsDomain)
DELETE old
WITH DISTINCT sn
SET sn.source_domains = $source_domains,
    sn.physics_domain = $physics_domain,
    sn.domain_decision_candidates = $decision_candidates,
    sn.domain_decision_choice = $decision_choice,
    sn.domain_decision_probability = $decision_probability,
    sn.domain_decision_model = $decision_model,
    sn.domain_decision_at = $decision_at,
    sn.domain_correction = $correction,
    sn.domain_correction_reason = $correction_reason
WITH sn
UNWIND $edge_domains AS domain_id
MERGE (d:PhysicsDomain {id: domain_id})
MERGE (sn)-[:HAS_PHYSICS_DOMAIN]->(d)
RETURN sn.id AS sn_id
"""


def _expected_source_domains(row: Mapping[str, Any]) -> frozenset[str]:
    """The domains a name's ``source_domains`` must equal.

    A name with DD sources takes their domains. A derived parent with none
    unions its live children's domains, taken from the children's DD sources
    where they have any and their stored ``source_domains`` otherwise.
    """
    own = {d for d in (row.get("own_domains") or []) if d}
    if own:
        return frozenset(own)
    union: set[str] = set()
    for d in row.get("child_dd_domains") or []:
        if d:
            union.add(d)
    for entry in row.get("child_stored_domains") or []:
        union |= {d for d in (entry or []) if d}
    return frozenset(union)


def write_name_domain(
    gc: Any,
    sn_id: str,
    decision: DomainDecision,
    *,
    correction: DomainCorrection | None = None,
) -> NameDomainWrite:
    """Set a name's ``physics_domain``, ``source_domains`` and edges together.

    ``source_domains`` is the name's current DD sources' domains (a derived
    parent's is its children's union). ``physics_domain`` is the recorded
    correction when one exists, otherwise the stored decision; an undecided
    name keeps its current value. The ``HAS_PHYSICS_DOMAIN`` edges are written
    to exactly one per entry of ``source_domains``. All of it is one
    transaction, so the three cannot disagree.
    """
    rows = [dict(row) for row in gc.query(_READ_STATE, sn_id=sn_id)]
    if not rows:
        raise ValueError(f"standard name {sn_id!r} is not in the graph")
    row = rows[0]
    source_domains = _expected_source_domains(row)
    current = row.get("physics_domain")
    if correction is not None:
        target = correction.domain
        reason = None
        correction_domain = correction.domain
        correction_reason = correction.reason
    else:
        correction_domain = None
        correction_reason = None
        if decision.decided:
            target = decision.choice
            reason = None
        else:
            target = current
            reason = decision.undecided_reason or "no stored decision"
    edge_domains = sorted(source_domains)
    gc.query(
        _WRITE_DOMAIN,
        sn_id=sn_id,
        source_domains=edge_domains,
        physics_domain=target,
        decision_candidates=list(decision.candidates),
        decision_choice=decision.choice,
        decision_probability=decision.probability,
        decision_model=decision.model,
        decision_at=decision.decided_at,
        correction=correction_domain,
        correction_reason=correction_reason,
        edge_domains=edge_domains,
    )
    return NameDomainWrite(
        sn_id=sn_id,
        physics_domain=target,
        source_domains=tuple(edge_domains),
        decided=correction is not None or decision.decided,
        correction=correction_domain,
        reason=reason,
    )
