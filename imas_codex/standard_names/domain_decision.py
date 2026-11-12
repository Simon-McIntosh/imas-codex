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
    "FamilyDomainDecision",
    "FamilyDomainOutcome",
    "NameDomainWrite",
    "all_domains",
    "band_domains",
    "band_of",
    "band_text",
    "band_umbrella",
    "build_family_band_question",
    "build_family_domain_question",
    "build_family_question",
    "build_family_state",
    "build_question",
    "build_state",
    "candidate_domains",
    "decide_and_write_family",
    "decide_domain",
    "decide_family_domain",
    "derived_parent_domain_decision",
    "domain_bands",
    "dominant_domain",
    "family_candidate_domains",
    "family_path_counts",
    "fan_out_family_decision",
    "ids_domain_map",
    "option_text",
    "read_family_decision",
    "store_family_decision",
    "undecided",
    "write_name_domain",
]

SEAT = "sn-domain-tree"
QUESTION = "physics_domain"
#: The first-stage Choice of the widened decision: which band the concept is in.
BAND_QUESTION = "physics_domain_band"
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


def all_domains() -> tuple[str, ...]:
    """Every domain the guidance defines, in precedence order."""
    return tuple(
        domain
        for band in _guidance().get("precedence") or ()
        for domain in band["domains"]
    )


def domain_bands() -> tuple[str, ...]:
    """Every band umbrella, in precedence order: the widened first-stage options."""
    return tuple(band["umbrella"] for band in _guidance().get("precedence") or ())


def band_domains(umbrella: str) -> tuple[str, ...]:
    """The domains one band holds, in precedence order, the umbrella among them."""
    for band in _guidance().get("precedence") or ():
        if band["umbrella"] == umbrella:
            return tuple(band["domains"])
    return ()


def band_text(umbrella: str) -> str:
    """One band's rationale, for the first-stage Choice."""
    for band in _guidance().get("precedence") or ():
        if band["umbrella"] == umbrella:
            return " ".join(str(band.get("rationale") or "").split())
    return ""


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


def undecided(name: str, options: Sequence[str], reason: str) -> DomainDecision:
    """The decision for a name whose Choice could not be made, with its reason.

    The candidates it was offered are kept, so a caller that stores the
    decision keeps what the name could have chosen among rather than erasing
    the record.
    """
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
        return undecided(name, options, "the Jev call returned no answer")
    choice = answer.get("choice")
    probabilities = {
        str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
    }
    probability = probabilities.get(str(choice))
    if choice not in options:
        return undecided(
            name, options, f"the choice {choice!r} is not one of the candidates"
        )
    if probability is None:
        return undecided(name, options, "the answer carried no probability")
    if probability < CONFIDENCE_FLOOR:
        return undecided(
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
    return asyncio.run(decide_domain(name, description, sources, ids_map, model=model))


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


def _ranked_domains(domains: Sequence[str]) -> dict[str, int]:
    """The domains these carry, most frequent first, ties broken by name."""
    return path_counts({str(index): domain for index, domain in enumerate(domains)})


def dominant_domain(domains: Sequence[str]) -> str | None:
    """The domain most of these carry, ties broken by name; ``None`` if none do."""
    return next(iter(_ranked_domains(domains)), None)


def derived_parent_domain_decision(
    decision: DomainDecision,
    child_domains: Sequence[str],
) -> DomainDecision:
    """The decision a derived parent carries: its stored choice, else its children's.

    A derived parent has no sources of its own, so its domain is the one its
    children carry. A stored choice wins, because a settled decision outranks
    an inherited one. Otherwise the domain the most children hold becomes the
    parent's choice, ties broken by name so the same child set always yields
    the same parent. A parent whose children carry no domain at all keeps the
    scalar it holds and stays undecided, so a decision is never invented where
    there is nothing to inherit.
    """
    if decision.decided:
        return decision
    ranked = _ranked_domains(child_domains)
    if not ranked:
        return decision
    chosen = next(iter(ranked))
    return DomainDecision(
        name=decision.name,
        candidates=tuple(sorted(ranked)),
        choice=chosen,
        probability=ranked[chosen] / sum(ranked.values()),
        model="derived-parent-majority",
        decided_at=None,
    )


def write_name_domain(
    gc: Any,
    sn_id: str,
    decision: DomainDecision,
    *,
    correction: DomainCorrection | None = None,
    fallback_domain: str | None = None,
) -> NameDomainWrite:
    """Set a name's ``physics_domain``, ``source_domains`` and edges together.

    ``source_domains`` is the name's current DD sources' domains (a derived
    parent's is its children's union). ``physics_domain`` is the recorded
    correction when one exists, otherwise the stored decision; an undecided
    name keeps its current value, or ``fallback_domain`` when it holds none —
    a successor minted by a rename carries no scalar yet and inherits its
    predecessor's. An undecided name that holds no scalar and whose sources
    agree on a single domain takes that domain, so its scalar, its source set
    and its edges agree; several source domains stay unset and undecided. The
    ``HAS_PHYSICS_DOMAIN`` edges are written to exactly one per entry of
    ``source_domains``. All of it is one transaction, so the three cannot
    disagree.
    """
    rows = [dict(row) for row in gc.query(_READ_STATE, sn_id=sn_id)]
    if not rows:
        raise ValueError(f"standard name {sn_id!r} is not in the graph")
    row = rows[0]
    source_domains = _expected_source_domains(row)
    current = row.get("physics_domain")
    if current is None:
        current = fallback_domain
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
    if target is None and len(source_domains) == 1:
        # An undecided name with no scalar of its own still has to agree with
        # its edges, so it takes the one domain its DD sources carry. The
        # sources agree on it, so this is derived rather than a guess; more
        # than one source domain is an ambiguity the Choice exists to settle,
        # so it stays unset and the name is reported undecided.
        target = next(iter(source_domains))
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


# ---------------------------------------------------------------------------
# One decision per family
# ---------------------------------------------------------------------------
#
# A name's domain describes the concept it names, and a concept is shared by
# every member of its family: X_density and krypton_density measure the same
# quantity, so they belong in the same domain. Deciding each name separately
# offers each only its own sources' domains, so two siblings sourced from
# different parts of the DD cannot agree. The family decision offers the union
# of every member's options, asks one Choice, stores it once, and writes it to
# every member through :func:`write_name_domain`.


@dataclass(frozen=True)
class FamilyDomainDecision:
    """The single domain decision a family carries, or why it is undecided."""

    family_id: str
    members: tuple[str, ...]
    candidates: tuple[str, ...]
    choice: str | None
    probability: float | None
    model: str | None
    decided_at: str | None
    undecided_reason: str | None = None
    corrected: bool = False
    cost: float = 0.0

    @property
    def decided(self) -> bool:
        return self.choice is not None and self.undecided_reason is None


@dataclass(frozen=True)
class FamilyDomainOutcome:
    """What one family's decision did: the record, the writes, the holdouts."""

    decision: FamilyDomainDecision
    writes: tuple[NameDomainWrite, ...]
    unwritten: tuple[str, ...]


def family_candidate_domains(
    members: Sequence[Mapping[str, Any]], ids_map: Mapping[str, str]
) -> tuple[str, ...]:
    """The domains every member's sources together supply for the family.

    Options are the union across members, so a member with one unusual source
    neither narrows nor widens the choice alone.
    """
    found: set[str] = set()
    for member in members:
        found |= set(candidate_domains(member.get("sources") or {}, ids_map))
    return tuple(sorted(found))


def family_path_counts(members: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Path-domain counts across the whole family, most frequent first."""
    merged: dict[str, str | None] = {}
    for index, member in enumerate(members):
        for path, domain in (member.get("sources") or {}).items():
            merged[f"{index}:{path}"] = domain
    return path_counts(merged)


def build_family_state(
    family_id: str,
    members: Sequence[Mapping[str, Any]],
    ids_map: Mapping[str, str],
    options: Sequence[str],
) -> dict[str, Any]:
    """The state a family's Choice is asked of, carrying every member's sources."""
    counts = family_path_counts(members)
    return {
        "family": family_id,
        "member_count": len(members),
        "members": [
            {
                "name": member.get("id") or "",
                "description": member.get("description") or "",
                "source_paths": [
                    {"path": path, "stored_domain": domain or "unclassified"}
                    for path, domain in sorted((member.get("sources") or {}).items())
                ],
            }
            for member in members
        ],
        "source_path_count_by_candidate_domain": {
            domain: counts.get(domain, 0) for domain in options
        },
        "source_path_count_by_band": _band_counts(counts),
    }


def _band_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """The family's source-path counts rolled up to band umbrellas."""
    banded: Counter[str] = Counter()
    for domain, count in counts.items():
        umbrella = band_umbrella(domain)
        if umbrella:
            banded[umbrella] += count
    return {band: banded.get(band, 0) for band in domain_bands()}


def build_family_band_question(member_count: int) -> dict[str, Any]:
    """The first stage of the widened Choice: which band the concept belongs to.

    The options are the band umbrellas rather than domains, so the model picks
    the subject area on a short list and the domain within it is chosen next.
    A flat choice over all thirty-two domains dilutes a confident answer below
    the confidence floor; the band split keeps each stage's option list short
    enough for a decided probability.
    """
    instructions = (
        f"Choose the one physics band that best describes this family of "
        f"{member_count} standard names, which measure the same quantity at "
        "the same locus and differ only in species, charge state, component or "
        "device. Judge what the quantity measures, not which IDS carries it: "
        "the family's source domains are evidence of where the data lives, not "
        "a bound on where the member names belong. Every band is offered, so a "
        "band none of the sources' domains falls in may be the right one. The "
        "physics of the plasma itself outranks the system that measures it or "
        "acts on it; choose general only when no narrower band fits."
    )
    order = list(domain_bands())
    return {
        BAND_QUESTION: {
            "type": "choice",
            "instructions": instructions,
            "criteria": {band: band_text(band) for band in order},
        }
    }


def build_family_domain_question(
    options: Sequence[str], member_count: int, *, widened: bool = True
) -> dict[str, Any]:
    """The second stage: which domain of the chosen band the family belongs to."""
    instructions = (
        f"Choose the one physics domain that best describes this family of "
        f"{member_count} standard names, which measure the same quantity at "
        "the same locus and differ only in species, charge state, component or "
        "device. Judge the concept the family names, not which IDS carries any "
        "one member's sources: every member receives the domain you choose. "
        "These are every domain of the band this family was judged to belong "
        "to, so a domain none of its sources’ paths lives in may be the right "
        "one: the sources' stored domains are evidence, not a bound. Prefer "
        "the most specific option that plainly fits the quantity; choose the "
        "band umbrella only when no narrower option fits. The options are "
        "listed most specific first."
        if widened
        else build_family_question(options, member_count)[QUESTION]["instructions"]
    )
    order = sorted(options, key=lambda d: (band_of(d) != "general", d))
    return {
        QUESTION: {
            "type": "choice",
            "instructions": instructions,
            "criteria": {domain: option_text(domain) for domain in order},
        }
    }


def build_family_question(options: Sequence[str], member_count: int) -> dict[str, Any]:
    """The Choice for a family: one domain for every name it holds.

    The question states the family is decided once for all its members, so the
    model judges the concept the members share rather than any one member's
    own sources.
    """
    instructions = (
        f"Choose the one physics domain that best describes this family of "
        f"{member_count} standard names, which measure the same quantity at "
        "the same locus and differ only in species, charge state, component or "
        "device. Judge the concept the family names, not which IDS carries "
        "any one member's sources: every member receives the domain you "
        "choose. Every option is a domain the Data Dictionary supplies for at "
        "least one member's source paths, the domain of one of their source "
        "IDSs, a broader band umbrella one of those falls under, or general. "
        "Prefer the most specific option that plainly fits the quantity; "
        "choose general only when no narrower option fits. The options are "
        "listed most specific first."
    )
    order = sorted(options, key=lambda d: (band_of(d) != "general", d))
    return {
        QUESTION: {
            "type": "choice",
            "instructions": instructions,
            "criteria": {domain: option_text(domain) for domain in order},
        }
    }


def _family_recorded_correction(
    members: Sequence[Mapping[str, Any]],
) -> str | None:
    """The domain a recorded lead correction gives the family, if one does.

    A correction recorded on any member applies to the family, because the
    family is one concept. Members whose corrections disagree leave the
    family to its own Choice rather than picking one arbitrarily.
    """
    recorded = {
        str(member["domain_correction"])
        for member in members
        if member.get("domain_correction")
    }
    return recorded.pop() if len(recorded) == 1 else None


def _answer_of(results: Any, key: str) -> Any:
    """The Choice answer for *key* in a ``decide_batch`` result, or ``None``."""
    if results and results[0]:
        answers = results[0][0]
        if isinstance(answers, dict):
            return answers.get(key)
    return None


def _probabilities(answer: Mapping[str, Any]) -> dict[str, float]:
    return {
        _pretty(str(k)): float(v)
        for k, v in (answer.get("probabilities") or {}).items()
    }


async def _decide_widened(
    family_id: str,
    member_ids: tuple[str, ...],
    members: Sequence[Mapping[str, Any]],
    ids_map: Mapping[str, str],
    model: str,
) -> FamilyDomainDecision:
    """The widened decision: which band, then which domain of that band.

    The option set is every domain of the band the concept is judged to belong
    to, not the domains the family's sources supply. A family sourced entirely
    from one part of the Data Dictionary can therefore receive a domain none of
    its sources lives in, which the source-bounded set makes impossible. The
    band stage keeps each Choice short enough that a confident answer stays
    above :data:`CONFIDENCE_FLOOR`; the stored probability is the band's
    probability times the domain's probability within the band, because both
    have to be right for the chosen domain to be right.
    """
    if not members:
        return FamilyDomainDecision(
            family_id=family_id,
            members=member_ids,
            candidates=(),
            choice=None,
            probability=None,
            model=model,
            decided_at=None,
            undecided_reason="the family has no members",
        )
    everything = all_domains()
    state = build_family_state(family_id, members, ids_map, everything)
    bands = domain_bands()

    def held(
        reason: str, cost: float, candidates: Sequence[str]
    ) -> FamilyDomainDecision:
        return FamilyDomainDecision(
            family_id=family_id,
            members=member_ids,
            candidates=tuple(candidates),
            choice=None,
            probability=None,
            model=model,
            decided_at=None,
            undecided_reason=reason,
            cost=cost,
        )

    band_results, band_cost = await decide_batch(
        [state],
        build_family_band_question(len(members)),
        model=model,
        service=SERVICE_TAG,
        concurrency=CONCURRENCY,
    )
    band_answer = _answer_of(band_results, BAND_QUESTION)
    if not isinstance(band_answer, dict):
        return held("the Jev call returned no band answer", band_cost, everything)
    band = band_answer.get("choice")
    band_probability = _probabilities(band_answer).get(_pretty(str(band)))
    if band not in bands:
        return held(
            f"the band {band!r} is not one of the candidates", band_cost, everything
        )
    if band_probability is None:
        return held("the band answer carried no probability", band_cost, everything)

    options = band_domains(str(band))
    cost = band_cost
    if len(options) == 1:
        choice, probability = options[0], band_probability
    else:
        results, domain_cost = await decide_batch(
            [state],
            build_family_domain_question(options, len(members)),
            model=model,
            service=SERVICE_TAG,
            concurrency=CONCURRENCY,
        )
        cost = band_cost + domain_cost
        answer = _answer_of(results, QUESTION)
        if not isinstance(answer, dict):
            return held("the Jev call returned no domain answer", cost, options)
        choice = answer.get("choice")
        chosen = _probabilities(answer).get(_pretty(str(choice)))
        if choice not in options:
            return held(
                f"the choice {choice!r} is not one of the candidates", cost, options
            )
        if chosen is None:
            return held("the domain answer carried no probability", cost, options)
        probability = band_probability * chosen
    if probability is None or probability < CONFIDENCE_FLOOR:
        return held(
            f"chosen probability {float(probability or 0.0):.3f} is below the floor "
            + f"{CONFIDENCE_FLOOR:.2f}",
            cost,
            options,
        )
    return FamilyDomainDecision(
        family_id=family_id,
        members=member_ids,
        candidates=options,
        choice=str(choice),
        probability=probability,
        model=model,
        decided_at=datetime.now(UTC).isoformat(),
        cost=cost,
    )


async def decide_family_domain(
    family_id: str,
    members: Sequence[Mapping[str, Any]],
    ids_map: Mapping[str, str],
    *,
    model: str,
    bounded_by_sources: bool = False,
) -> FamilyDomainDecision:
    """Ask Jev one domain Choice for the whole family and return the decision.

    The options are every domain of the band the concept is judged to belong
    to, not the domains its sources supply: the sources' stored domains are
    evidence, and a domain none of them lives in can be chosen. Passing
    ``bounded_by_sources=True`` restricts the options to the family's source
    domains as the earlier option set did; it exists to measure the two sets
    against each other in one run and is not the shipped behaviour.

    A family whose options collapse to one domain takes it without a call. A
    recorded lead correction wins over the Choice and applies to the whole
    family. A call that fails, or a choice below :data:`CONFIDENCE_FLOOR`,
    leaves the family undecided and names the reason: the caller writes none
    of its members.
    """
    member_ids = tuple(str(member.get("id")) for member in members)
    options = family_candidate_domains(members, ids_map)
    corrected_domain = _family_recorded_correction(members)
    if corrected_domain is not None:
        return FamilyDomainDecision(
            family_id=family_id,
            members=member_ids,
            candidates=options,
            choice=corrected_domain,
            probability=1.0,
            model="recorded-correction",
            decided_at=datetime.now(UTC).isoformat(),
            corrected=True,
        )
    if not bounded_by_sources:
        return await _decide_widened(family_id, member_ids, members, ids_map, model)
    if len(options) <= 1:
        return FamilyDomainDecision(
            family_id=family_id,
            members=member_ids,
            candidates=options,
            choice=options[0] if options else None,
            probability=1.0,
            model=model,
            decided_at=datetime.now(UTC).isoformat(),
        )
    results, cost = await decide_batch(
        [build_family_state(family_id, members, ids_map, options)],
        build_family_question(options, len(members)),
        model=model,
        service=SERVICE_TAG,
        concurrency=CONCURRENCY,
    )
    answer: dict[str, Any] | None = None
    if results and results[0]:
        answers = results[0][0]
        if isinstance(answers, dict):
            answer = answers.get(QUESTION)

    def held(reason: str) -> FamilyDomainDecision:
        return FamilyDomainDecision(
            family_id=family_id,
            members=member_ids,
            candidates=options,
            choice=None,
            probability=None,
            model=model,
            decided_at=None,
            undecided_reason=reason,
            cost=cost,
        )

    if not isinstance(answer, dict):
        return held("the Jev call returned no answer")
    choice = answer.get("choice")
    probabilities = {
        str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
    }
    probability = probabilities.get(str(choice))
    if choice not in options:
        return held(f"the choice {choice!r} is not one of the candidates")
    if probability is None:
        return held("the answer carried no probability")
    if probability < CONFIDENCE_FLOOR:
        return held(
            f"chosen probability {probability:.3f} is below the floor "
            + f"{CONFIDENCE_FLOOR:.2f}"
        )
    return FamilyDomainDecision(
        family_id=family_id,
        members=member_ids,
        candidates=options,
        choice=str(choice),
        probability=probability,
        model=model,
        decided_at=datetime.now(UTC).isoformat(),
        cost=cost,
    )


#: One record per family, holding the decision every member reads. The family
#: node owns the decision; the name owns only the id of the family it reads, so
#: the members of a family cannot disagree about the domain.
_STORE_FAMILY_DECISION = """
MERGE (f:StandardNameDomainFamily {id: $family_id})
SET f.candidates = $candidates,
    f.choice = $choice,
    f.probability = $probability,
    f.model = $model,
    f.decided_at = $decided_at,
    f.member_count = $member_count,
    f.undecided_reason = $undecided_reason
WITH f
UNWIND $members AS member_id
MATCH (sn:StandardName {id: member_id})
MERGE (sn)-[:IN_DOMAIN_FAMILY]->(f)
SET sn.domain_family_id = $family_id
RETURN count(*) AS linked
"""

#: The record a member reads: its family's single decision.
_READ_FAMILY_DECISION = """
MATCH (sn:StandardName {id: $sn_id})-[:IN_DOMAIN_FAMILY]->(f:StandardNameDomainFamily)
RETURN f.id AS family_id,
       f.candidates AS candidates,
       f.choice AS choice,
       f.probability AS probability,
       f.model AS model,
       f.decided_at AS decided_at,
       f.undecided_reason AS undecided_reason
"""


def store_family_decision(gc: Any, decision: FamilyDomainDecision) -> int:
    """Store the family's decision once and point every member at the record.

    Returns the number of members linked. A member the graph does not hold is
    not linked; the record itself is still stored, so a later read finds it.
    """
    rows = list(
        gc.query(
            _STORE_FAMILY_DECISION,
            family_id=decision.family_id,
            candidates=list(decision.candidates),
            choice=decision.choice,
            probability=decision.probability,
            model=decision.model,
            decided_at=decision.decided_at,
            member_count=len(decision.members),
            undecided_reason=decision.undecided_reason,
            members=list(decision.members),
        )
        or []
    )
    return int(rows[0].get("linked") or 0) if rows else 0


def read_family_decision(gc: Any, sn_id: str) -> FamilyDomainDecision | None:
    """The single stored decision *sn_id* reads, or ``None`` when it has none."""
    rows = list(gc.query(_READ_FAMILY_DECISION, sn_id=sn_id) or [])
    if not rows:
        return None
    row = dict(rows[0])
    return FamilyDomainDecision(
        family_id=str(row.get("family_id") or ""),
        members=(),
        candidates=tuple(row.get("candidates") or ()),
        choice=row.get("choice"),
        probability=row.get("probability"),
        model=row.get("model"),
        decided_at=row.get("decided_at"),
        undecided_reason=row.get("undecided_reason"),
    )


def fan_out_family_decision(
    gc: Any, decision: FamilyDomainDecision
) -> tuple[NameDomainWrite, ...]:
    """Write the family's one decision to every member through the one writer.

    An undecided family writes nothing: the caller names every member in the
    outcome's ``unwritten`` list instead. A corrected family writes the
    correction to every member, so a recorded correction applies to the family.
    """
    if not decision.decided:
        return ()
    correction = (
        DomainCorrection(
            domain=str(decision.choice),
            reason="recorded lead correction for the family",
        )
        if decision.corrected
        else None
    )
    writes: list[NameDomainWrite] = []
    for member_id in decision.members:
        writes.append(
            write_name_domain(
                gc,
                member_id,
                DomainDecision(
                    name=member_id,
                    candidates=decision.candidates,
                    choice=decision.choice,
                    probability=decision.probability,
                    model=decision.model,
                    decided_at=decision.decided_at,
                ),
                correction=correction,
            )
        )
    return tuple(writes)


def decide_and_write_family(
    gc: Any,
    sn_id: str,
    ids_map: Mapping[str, str],
    *,
    model: str,
    family: Mapping[str, Any] | None = None,
) -> FamilyDomainOutcome:
    """Decide one family's domain once and write it to every member.

    *family* is the read from
    :func:`imas_codex.standard_names.families.fetch_domain_family` for
    *sn_id*, and is read when not supplied. This is the entry point a caller
    needs: option union across the members, one judgment, one stored record,
    and a fan-out through :func:`write_name_domain`.
    """
    if family is None:
        from imas_codex.standard_names.families import fetch_domain_family

        family = fetch_domain_family(sn_id, gc)
    members = list(family.get("members") or [])
    family_id = str(family.get("key") or "")
    decision = asyncio.run(
        decide_family_domain(family_id, members, ids_map, model=model)
    )
    store_family_decision(gc, decision)
    writes = fan_out_family_decision(gc, decision)
    written = {entry.sn_id for entry in writes}
    return FamilyDomainOutcome(
        decision=decision,
        writes=writes,
        unwritten=tuple(
            member["id"] for member in members if member["id"] not in written
        ),
    )
