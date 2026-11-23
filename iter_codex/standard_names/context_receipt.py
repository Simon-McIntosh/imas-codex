"""A record of the context each model-calling stage was given.

A standard-name stage is only as sound as the context it receives, and until
now nothing in the pipeline could state what a stage was shown. The case file
(:mod:`iter_codex.standard_names.case_file`) already assembles the whole of
what a stage may read about one quantity and judges its own completeness; this
module turns that record into a per-stage receipt a running stage writes for
each name it handles, and into a census that renders each stage's prompt for a
name without sending it.

A receipt names, per channel the stage is expected to receive, whether the
channel arrived and its size, and carries a digest of the rendered prompt. A
stage whose expected channel is absent records the omission by name and does
not report the context complete. The channels are the case file's own, so
there is no second notion of "expected" to drift from the first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from iter_codex.standard_names.case_file import CaseFile

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA = "iter-codex.standard-name-context-receipt"
RECEIPT_VERSION = 1


class StagePromptError(RuntimeError):
    """A stage's production template could not render from the context given.

    Raised rather than substituted with a marker: a receipt is evidence about
    what a stage was shown, and a digest over a placeholder is evidence about
    nothing.
    """


#: The model-calling stages, in pipeline order. A stage writes one receipt per
#: name it handles; the census reads the same set so its ranks line up.
STAGES: tuple[str, ...] = (
    "extraction",
    "name_composition",
    "name_review",
    "refinement",
    "documentation",
    "parent_enrichment",
    "scoring_review",
)

#: Every channel a receipt can name. The last two are the parent's children
#: channel and the producer channel, which states whether a producer exists at
#: all so a name with no source is reported as such rather than as an empty
#: source. ``coordinates`` is held here because the case file carries it and a
#: future template may read it; no current stage template does, so it appears
#: in no stage's expectations and the census reports that.
CHANNELS: tuple[str, ...] = (
    "source_text",
    "enclosing_structure_text",
    "unit",
    "coordinates",
    "family",
    "neighbours",
    "incumbent",
    "grammar",
    "vocabulary",
    "children",
    "producer",
)

#: The channels each stage is expected to receive. A channel is expected when
#: the stage's prompt template reads it: composition reads the source text, the
#: enclosing structures and the section's family; parent enrichment reads the
#: children and the incumbent.
STAGE_CHANNELS: dict[str, tuple[str, ...]] = {
    "extraction": (
        "producer",
        "source_text",
        "enclosing_structure_text",
        "unit",
    ),
    "name_composition": (
        "source_text",
        "enclosing_structure_text",
        "unit",
        "family",
        "neighbours",
        "grammar",
        "vocabulary",
    ),
    "name_review": (
        "source_text",
        "enclosing_structure_text",
        "incumbent",
        "family",
        "neighbours",
        "grammar",
        "vocabulary",
    ),
    "refinement": (
        "incumbent",
        "source_text",
        "enclosing_structure_text",
        "neighbours",
        "grammar",
    ),
    "documentation": (
        "source_text",
        "enclosing_structure_text",
        "unit",
        "family",
        "neighbours",
        "grammar",
    ),
    "parent_enrichment": ("children", "incumbent"),
    "scoring_review": (
        "incumbent",
        "source_text",
        "enclosing_structure_text",
        "family",
        "neighbours",
        "grammar",
        "vocabulary",
    ),
}

#: The prompt template pair a stage renders for one name, ``(system, user)``.
#: ``None`` when the stage renders no system prompt. These are the production
#: templates; the census renders them and sends nothing.
STAGE_TEMPLATES: dict[str, tuple[str | None, str]] = {
    "extraction": ("sn/self_refine_system", "sn/self_refine_user"),
    "name_composition": ("sn/generate_name_system", "sn/generate_name_dd"),
    "name_review": ("sn/review_names_system", "sn/review_names_user"),
    "refinement": ("sn/refine_name_system", "sn/refine_name_user"),
    "documentation": ("sn/generate_docs_system", "sn/generate_docs_user"),
    "parent_enrichment": ("sn/enrich_parent_system", "sn/enrich_parent_user"),
    "scoring_review": ("sn/review_docs_system", "sn/review_docs_user"),
}

#: Stages whose children channel is expected for a *derived parent* only. A
#: name that is not a parent has no children, and a "none" answer is complete.
_PARENT_CHILD_STAGES: frozenset[str] = frozenset(
    {"extraction", "name_composition", "documentation", "scoring_review"}
)


def _joined(value: Any) -> str:
    """Return *value* as one string, whatever its shape."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def is_parent(content: Mapping[str, Any]) -> bool:
    """A name is a parent when it has live children."""
    return str((content.get("children") or {}).get("status") or "") == "members"


def _source_text(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    sources = list(content.get("sources") or ())
    if not sources:
        return False, 0, "no source is bound to this name"
    missing = [s.get("id") for s in sources if not str(s.get("text") or "").strip()]
    size = sum(len(str(s.get("text") or "")) for s in sources)
    if missing:
        return (
            False,
            size,
            f"source text absent for {len(missing)} of {len(sources)} source(s)",
        )
    return True, size, f"{len(sources)} source(s) with text"


def _enclosing_text(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    sources = list(content.get("sources") or ())
    if not sources:
        return False, 0, "no source is bound to this name"
    absent = [
        s.get("id")
        for s in sources
        if str((s.get("enclosing") or {}).get("status") or "") != "ok"
    ]
    size = sum(
        len(str((s.get("enclosing") or {}).get("statement") or "")) for s in sources
    )
    if absent:
        return (
            False,
            size,
            f"enclosing text absent for {len(absent)} of {len(sources)} source(s)",
        )
    return True, size, f"enclosing structures present for {len(sources)} source(s)"


def _unit(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    units = [str(s.get("unit")) for s in content.get("sources") or () if s.get("unit")]
    incumbent = content.get("incumbent") or {}
    if incumbent.get("unit"):
        units.append(str(incumbent.get("unit")))
    if units:
        return True, len(" ".join(units)), f"{len(units)} unit(s) declared"
    return False, 0, "no unit is declared by any source or by the incumbent"


def _coordinates(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    coords = [
        str(coordinate)
        for source in content.get("sources") or ()
        for coordinate in source.get("coordinates") or ()
    ]
    if coords:
        return True, len(" ".join(coords)), f"{len(coords)} coordinate reference(s)"
    return False, 0, "no source declares coordinates"


def _family(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    block = content.get("family") or {}
    status = str(block.get("status") or "")
    statement = str(block.get("statement") or "").strip()
    if status in {"members", "none"} and statement:
        members = list(block.get("members") or ())
        return (
            True,
            len(_joined(block)),
            f"family answered {status} ({len(members)} member(s))",
        )
    return False, 0, statement or status or "family channel unread"


def _neighbours(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    block = content.get("neighbours") or {}
    parts = []
    absent = []
    for channel in ("by_subject", "by_locus"):
        sub = block.get(channel) or {}
        status = str(sub.get("status") or "")
        if status in {"ok", "none"} and str(sub.get("statement") or "").strip():
            parts.append(sub.get("statement"))
        else:
            absent.append(channel)
    size = len(_joined(parts))
    if absent:
        return False, size, f"neighbours absent for {len(absent)} of 2 channel(s)"
    return True, size, "subject and locus neighbours answered"


def _incumbent(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    incumbent = content.get("incumbent")
    if not incumbent:
        return False, 0, "no incumbent name is recorded"
    return (
        True,
        len(_joined(incumbent)),
        "incumbent name, definition and documentation present",
    )


def _grammar(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    block = content.get("grammar") or {}
    status = str(block.get("status") or "")
    if status in {"parsed", "unparsed", "not_a_name"}:
        return (
            True,
            len(_joined(block.get("segments") or {})),
            f"grammar answered {status}",
        )
    return False, 0, str(block.get("statement") or status or "grammar channel unread")


def _vocabulary(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    block = (content.get("grammar") or {}).get("vocabulary") or {}
    status = str(block.get("status") or "")
    if status in {"ok", "none"}:
        return (
            True,
            len(_joined(block.get("entries") or [])),
            f"vocabulary answered {status}",
        )
    return (
        False,
        0,
        str(block.get("statement") or status or "vocabulary channel unread"),
    )


def _children(content: Mapping[str, Any], *, is_parent: bool) -> tuple[bool, int, str]:
    block = content.get("children")
    if block is None:
        return False, 0, "the case file carries no children channel"
    status = str(block.get("status") or "")
    members = list(block.get("members") or ())
    size = len(_joined(block))
    if is_parent:
        if status == "members" and members:
            return True, size, f"{len(members)} child name(s)"
        return False, size, "a derived parent carries no child name"
    if status in {"members", "none"}:
        return True, size, f"children answered {status}"
    return (
        False,
        size,
        str(block.get("statement") or status or "children channel unread"),
    )


def _producer(content: Mapping[str, Any]) -> tuple[bool, int, str]:
    block = content.get("producer")
    if block is None:
        return False, 0, "the case file carries no producer channel"
    status = str(block.get("status") or "")
    if status in {"dd", "derived", "none"}:
        return True, len(_joined(block)), f"producer answered {status}"
    return False, 0, str(block.get("statement") or status or "producer channel unread")


_PROBES = {
    "source_text": _source_text,
    "enclosing_structure_text": _enclosing_text,
    "unit": _unit,
    "coordinates": _coordinates,
    "family": _family,
    "neighbours": _neighbours,
    "incumbent": _incumbent,
    "grammar": _grammar,
    "vocabulary": _vocabulary,
    "producer": _producer,
}


def _probe(
    channel: str, content: Mapping[str, Any], *, parent: bool
) -> tuple[bool, int, str]:
    if channel == "children":
        return _children(content, is_parent=parent)
    probe = _PROBES.get(channel)
    if probe is None:
        return False, 0, f"no probe is defined for channel {channel!r}"
    return probe(content)


class ChannelReceipt(BaseModel):
    """What one channel answered: whether it arrived and how much it carried."""

    model_config = ConfigDict(extra="forbid")

    present: bool
    size: int
    reason: str


class ContextReceipt(BaseModel):
    """The context one stage was given for one name."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str = RECEIPT_SCHEMA
    version: int = RECEIPT_VERSION
    stage: str
    name: str
    expected: list[str]
    channels: dict[str, ChannelReceipt]
    missing: list[str]
    complete: bool
    withheld: list[str]
    prompt_digest: str
    prompt_chars: int


def expected_channels(stage: str, content: Mapping[str, Any]) -> list[str]:
    """Return the channels *stage* is expected to receive for this name.

    The stage's template channels, plus the children channel when the name is a
    derived parent and the stage grounds on children.
    """
    channels = list(STAGE_CHANNELS[stage])
    if (
        stage in _PARENT_CHILD_STAGES
        and is_parent(content)
        and "children" not in channels
    ):
        channels.append("children")
    return channels


def prompt_digest(text: str) -> str:
    """Return the digest a receipt carries for a rendered prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def receipt_from_content(
    stage: str,
    name: str,
    content: Mapping[str, Any],
    *,
    prompt_text: str = "",
    withhold: Sequence[str] = (),
) -> ContextReceipt:
    """Build the receipt *stage* writes for *name* from its case-file content.

    ``withhold`` names channels to force absent, which is how the section's
    negative control shows a stage reporting the omission it is given.
    """
    if stage not in STAGE_CHANNELS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    unknown = sorted(set(withhold) - set(CHANNELS))
    if unknown:
        raise ValueError(
            f"unknown withheld channel(s) {unknown}; expected a subset of {list(CHANNELS)}"
        )
    parent = is_parent(content)
    expected = expected_channels(stage, content)
    channels: dict[str, ChannelReceipt] = {}
    for channel in expected:
        if channel in withhold:
            channels[channel] = ChannelReceipt(
                present=False,
                size=0,
                reason="withheld on purpose for the negative control",
            )
            continue
        present, size, reason = _probe(channel, content, parent=parent)
        channels[channel] = ChannelReceipt(present=present, size=size, reason=reason)
    missing = [channel for channel in expected if not channels[channel].present]
    return ContextReceipt(
        stage=stage,
        name=name,
        expected=expected,
        channels=channels,
        missing=missing,
        complete=not missing,
        withheld=sorted(set(withhold)),
        prompt_digest=prompt_digest(prompt_text) if prompt_text else "",
        prompt_chars=len(prompt_text),
    )


def receipt_from_case_file(
    stage: str,
    case_file: CaseFile | Mapping[str, Any],
    *,
    prompt_text: str = "",
    withhold: Sequence[str] = (),
) -> ContextReceipt:
    """Build a receipt from a case file record or its dumped mapping."""
    if isinstance(case_file, Mapping):
        content = dict(case_file.get("content") or {})
        target = dict(case_file.get("target") or {})
    else:
        content = dict(case_file.content)
        target = dict(case_file.target)
    name = str(target.get("id") or (content.get("target") or {}).get("id") or "")
    return receipt_from_content(
        stage, name, content, prompt_text=prompt_text, withhold=withhold
    )


def summarise(receipts: Iterable[ContextReceipt]) -> dict[str, Any]:
    """Count, per stage and channel, expected / present / absent over a cohort.

    A receipt carries only the channels its stage expected, so a channel that
    was not asked for is not counted; the shares are over the names that
    expected it. The shares are rounded so a re-read reproduces the file.
    """
    counts: dict[str, dict[str, list[int]]] = {}
    names: dict[str, int] = {}
    for receipt in receipts:
        names[receipt.stage] = names.get(receipt.stage, 0) + 1
        bucket = counts.setdefault(receipt.stage, {})
        for channel in receipt.expected:
            row = bucket.setdefault(channel, [0, 0, 0])
            row[0] += 1
            if receipt.channels[channel].present:
                row[1] += 1
            else:
                row[2] += 1
    table: dict[str, dict[str, dict[str, Any]]] = {}
    omissions: list[dict[str, Any]] = []
    for stage in STAGES:
        if stage not in counts:
            continue
        table[stage] = {}
        for channel in CHANNELS:
            if channel not in counts[stage]:
                continue
            expected, present, absent = counts[stage][channel]
            table[stage][channel] = {
                "expected": expected,
                "present": present,
                "absent": absent,
                "share_present": round(present / expected, 4) if expected else None,
                "share_absent": round(absent / expected, 4) if expected else None,
            }
            if absent:
                omissions.append(
                    {
                        "stage": stage,
                        "channel": channel,
                        "names": absent,
                        "expected": expected,
                        "share_absent": round(absent / expected, 4),
                    }
                )
    omissions.sort(key=lambda row: (-row["names"], row["stage"], row["channel"]))
    return {"names": names, "table": table, "omissions": omissions}


def _shared_context() -> dict[str, Any]:
    """The section-level prompt context the compose pipeline renders from.

    Grammar, vocabulary, field guidance and every other static block are built
    by :func:`iter_codex.standard_names.context.build_compose_context`, the call
    each live stage makes. This module asks that owner rather than supplying its
    own set: a thinner second set here is how three templates came to be
    reported as rendered while they were raising on a census key the pipeline
    does supply.
    """
    from iter_codex.standard_names.context import build_compose_context

    try:
        return dict(build_compose_context())
    except Exception:  # noqa: BLE001 - reported, and the render then fails loudly
        logger.warning(
            "context_receipt: the compose context is unavailable, so a census "
            "prompt would carry no grammar or field guidance",
            exc_info=True,
        )
        return {}


def _name_context(content: Mapping[str, Any]) -> dict[str, Any]:
    """The per-name half of a stage's prompt context, from a case file.

    Every key is read from the block that owns it. The family keeps the parent,
    the anchor and each member's variation axis, which is what the case file
    recorded from :func:`iter_codex.standard_names.context.fetch_sibling_family`;
    the two neighbour channels stay distinct rather than one standing in for
    both, and enclosure comes from the sources' own statements.

    These are the keys :func:`content_from_prompt_context` reads back out of a
    live stage's context and out of a receipt alike, so the two directions
    describe one prompt instead of two shapes that drift apart.
    """
    target = dict(content.get("target") or {})
    incumbent = dict(content.get("incumbent") or {})
    sources = list(content.get("sources") or ())
    first = dict(sources[0]) if sources else {}
    unit = incumbent.get("unit") or first.get("unit")
    family = dict(content.get("family") or {})
    members = list(family.get("members") or ())
    neighbours = dict(content.get("neighbours") or {})
    children = list((content.get("children") or {}).get("members") or ())
    grammar = dict(content.get("grammar") or {})
    name = incumbent.get("name") or target.get("id") or ""
    enclosing = [
        {"path": statement.get("path"), "text": statement.get("text")}
        for source in sources
        for statement in ((source.get("enclosing") or {}).get("statements") or ())
    ]
    item = {
        "id": incumbent.get("id") or target.get("id"),
        "name": name,
        "description": incumbent.get("description") or "",
        "documentation": incumbent.get("documentation") or "",
        "unit": unit,
        "kind": incumbent.get("kind") or target.get("kind"),
        "physics_domain": incumbent.get("physics_domain"),
        "coordinates": list(first.get("coordinates") or ()),
        "source_paths": [str(source.get("id")) for source in sources],
        "dd_description": " ".join(str(source.get("text") or "") for source in sources),
        "dd_documentation": " ".join(
            str(source.get("text") or "") for source in sources
        ),
        "dd_units": unit,
        "derived_children": [str(child.get("name") or "") for child in children],
    }
    sibling_family = {
        "parent": family.get("parent"),
        "anchor": family.get("anchor"),
        "siblings": [
            {
                "name": member.get("name"),
                "operator_kind": member.get("differs_on"),
                "axis": member.get("axis"),
                "description": member.get("description") or "",
            }
            for member in members
        ],
    }
    item["sibling_family"] = sibling_family

    def prompt_row(entry: Mapping[str, Any]) -> dict[str, Any]:
        """One case-file neighbour row in the shape a stage's prompt reads.

        The case file keys a neighbour by ``name`` and always carries ``score``,
        which is null for the same-base channel because that lookup returns no
        similarity. A stage's prompt entry is keyed ``id`` and carries ``score``
        only where there is one, so the null-scored case must drop the key
        rather than pass a null the template would format.
        """
        row: dict[str, Any] = {
            "id": entry.get("name"),
            "name": entry.get("name"),
            "description": entry.get("description") or "",
            "unit": entry.get("unit"),
        }
        if entry.get("score") is not None:
            row["score"] = entry["score"]
        return row

    by_subject = dict(neighbours.get("by_subject") or {})
    subject_rows = [dict(row) for row in (by_subject.get("members") or ())]
    same_base_source = "same physical base"
    vector_neighbours = [
        prompt_row(row) for row in subject_rows if row.get("source") != same_base_source
    ]
    same_base_neighbours = [
        prompt_row(row) for row in subject_rows if row.get("source") == same_base_source
    ]
    by_locus = dict(neighbours.get("by_locus") or {})
    locus_rows = [dict(row) for row in (by_locus.get("members") or ())]
    dd_description = item["dd_description"] or item["description"]
    dd_context = {
        "path": str(first.get("id") or ""),
        "ids_name": name,
        "unit": unit,
        "physics_domain": incumbent.get("physics_domain") or "",
        "description": item["description"] or dd_description,
        "documentation": dd_description,
        "coordinates": item["coordinates"],
        "ancestors": enclosing,
    }
    return {
        "item": item,
        "name": name,
        "description": item["description"],
        "segments": dict(grammar.get("segments") or {}),
        "dd_context": dd_context,
        # The diagnostics list is produced at composition time by the stage
        # that has just parsed the candidate. A census renders the prompt
        # without one, and the template states that absence rather than this
        # module inventing a diagnostic the stage did not have.
        "diagnostics": [],
        "anc": enclosing,
        "children": [str(child.get("name") or "") for child in children],
        "sib": [str(member.get("name") or "") for member in members],
        "cl": [str(member.get("name") or "") for member in members],
        "sibling_family": sibling_family,
        "vector_neighbours": vector_neighbours,
        "same_base_neighbours": same_base_neighbours,
        "same_path_neighbours": locus_rows,
        "locus_context": locus_rows or None,
    }


def _stage_context(content: Mapping[str, Any]) -> dict[str, Any]:
    """The full prompt context a stage's template reads for one name.

    The section-level half comes from the compose pipeline's own builder and
    the per-name half from the case file; nothing here restates either. A key
    neither can supply stays absent, and the render raises rather than passing
    silently, so a census cannot report a prompt the stage never saw.
    """
    return {**_shared_context(), **_name_context(content)}


def render_stage_prompt(
    stage: str,
    content: Mapping[str, Any],
    *,
    prompts_dir: Any = None,
) -> str:
    """Render *stage*'s prompt for one name from a case file, sending nothing.

    The system and user templates are the production templates and the context
    is the one the stage reads, so the returned text is what the stage would
    have been shown. A template that cannot render raises
    :class:`StagePromptError` naming the stage, the template and the missing
    variable: a receipt whose digest covered a placeholder instead would say a
    context arrived that the stage never had, and the census would report a
    prompt that does not exist.
    """
    if stage not in STAGE_TEMPLATES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    from iter_codex.llm.prompt_loader import render_prompt

    system_name, user_name = STAGE_TEMPLATES[stage]
    context = _stage_context(content)
    parts: list[str] = []
    for template in (system_name, user_name):
        if not template:
            continue
        try:
            parts.append(render_prompt(template, context, prompts_dir=prompts_dir))
        except Exception as error:
            raise StagePromptError(
                f"{stage}: template {template!r} did not render: {error}"
            ) from error
    return "\n\n".join(parts)


def receipt_for_stage(
    stage: str,
    case_file: CaseFile | Mapping[str, Any],
    *,
    withhold: Sequence[str] = (),
    prompts_dir: Any = None,
) -> ContextReceipt:
    """Render *stage*'s prompt and build its receipt from one case file."""
    if isinstance(case_file, Mapping):
        content = dict(case_file.get("content") or {})
        target = dict(case_file.get("target") or {})
    else:
        content = dict(case_file.content)
        target = dict(case_file.target)
    name = str(target.get("id") or (content.get("target") or {}).get("id") or "")
    prompt = render_stage_prompt(stage, content, prompts_dir=prompts_dir)
    return receipt_from_content(
        stage, name, content, prompt_text=prompt, withhold=withhold
    )


#: The sink a running stage appends its receipts to when nothing overrides it.
#: It follows the CLI's own log directory, so a stage's record exists without
#: the pipeline being configured to produce one — a receipt that is only
#: written when an environment variable is set is a receipt no pipeline runs
#: with, which is what this default exists to stop.
DEFAULT_RECEIPT_SINK = Path("~/.local/share/iter-codex/logs/context-receipts.jsonl")

#: Set this to redirect receipts, for a test or a stage run outside the CLI.
RECEIPT_SINK_ENV = "ITER_CODEX_CONTEXT_RECEIPT_SINK"


def receipt_sink() -> Path:
    """Return the path a receipt was appended to.

    ``ITER_CODEX_CONTEXT_RECEIPT_SINK`` wins when it is set; otherwise the
    module's own log file is used, so no stage has to be told where to write.
    """
    raw = os.environ.get(RECEIPT_SINK_ENV)
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_RECEIPT_SINK.expanduser()


def record_receipt(receipt: ContextReceipt, *, sink: Path | None = None) -> Path:
    """Append one receipt to the JSONL sink and return the path it went to.

    A receipt records what a stage was shown, so it is written on every run
    rather than when someone has configured it. The write is one ``O_APPEND``
    line per receipt, which is what lets concurrent stages share the file.

    A sink that cannot be written is reported at WARNING and does not raise:
    the receipt is observation, and failing a paid stage's turn because its
    recording surface was read-only would make the observation more dangerous
    than its absence. It is logged rather than passed over, because a silent
    skip is indistinguishable from a pipeline in which no stage ever ran.
    """
    path = sink or receipt_sink()
    line = json.dumps(receipt.model_dump(), sort_keys=True, ensure_ascii=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        logger.warning(
            "context_receipt: could not write the %s receipt for %r to %s",
            receipt.stage,
            receipt.name,
            path,
            exc_info=True,
        )
    return path


def emit_receipt(
    stage: str,
    name: str,
    content: Mapping[str, Any],
    *,
    prompt_text: str = "",
    sink: Path | None = None,
) -> ContextReceipt:
    """Build and record a stage receipt for one name in one call."""
    receipt = receipt_from_content(stage, name, content, prompt_text=prompt_text)
    record_receipt(receipt, sink=sink)
    return receipt


def content_from_prompt_context(
    context: Mapping[str, Any], *, name: str = ""
) -> dict[str, Any]:
    """Express a live stage's prompt context in the case file's channel shape.

    A stage assembles its own prompt context and sends that; the receipt's
    channels and its completeness are the case file's. This maps the one onto
    the other, so a stage records what it actually sent while the expectations
    stay a single table. A value the context does not carry is left absent and
    reported as such by the same probes the census uses.
    """
    item = dict(context.get("item") or {})
    if not item:
        items = list(context.get("items") or ())
        item = dict(items[0]) if items else {}
    name = str(item.get("id") or name)

    enclosing = list(
        context.get("anc")
        or (context.get("dd_context") or {}).get("ancestors")
        or item.get("enclosing")
        or item.get("ancestors")
        or ()
    )
    dd_text = str(item.get("dd_description") or item.get("description") or "")
    source = {
        "id": name,
        "exists": bool(dd_text),
        "text": dd_text,
        "unit": item.get("unit") or item.get("dd_units"),
        "coordinates": list(item.get("coordinates") or ()),
        "enclosing": {
            "status": "ok" if enclosing else "none",
            "statement": (
                f"{len(enclosing)} enclosing statement(s)"
                if enclosing
                else "no enclosing statement"
            ),
            "statements": enclosing,
        },
    }

    family = dict(context.get("sibling_family") or {})
    siblings = list(family.get("siblings") or ())
    if not siblings:
        siblings = [{"name": str(s)} for s in (context.get("sib") or ())]

    def _neighbour(block: Any) -> dict[str, Any]:
        members = list(block or ())
        return {
            "status": "ok" if members else "none",
            "statement": (
                f"{len(members)} neighbour(s)" if members else "no neighbour"
            ),
            "members": members,
        }

    grammar_block = context.get("grammar")
    vocabulary_sections = list(context.get("vocabulary_sections") or ())
    children = list(item.get("derived_children") or context.get("children") or ())
    return {
        "target": {"id": name, "kind": item.get("kind")},
        "incumbent": item or None,
        "sources": [source] if name else [],
        "family": {
            "status": "members" if siblings else "none",
            "statement": (f"{len(siblings)} sibling(s)" if siblings else "no family"),
            "members": siblings,
        },
        "neighbours": {
            "by_subject": _neighbour(context.get("vector_neighbours")),
            "by_locus": _neighbour(
                context.get("same_path_neighbours") or context.get("locus_context")
            ),
        },
        "grammar": {
            "status": "parsed" if grammar_block else "unread",
            "statement": "grammar contract supplied" if grammar_block else "no grammar",
            "segments": grammar_block or {},
            "vocabulary": {
                "status": "ok" if vocabulary_sections else "none",
                "statement": (
                    f"{len(vocabulary_sections)} vocabulary section(s)"
                    if vocabulary_sections
                    else "no vocabulary"
                ),
                "entries": vocabulary_sections,
            },
        },
        "children": {
            "status": "members" if children else "none",
            "statement": (
                f"{len(children)} child name(s)" if children else "no child name"
            ),
            "members": [{"name": str(child)} for child in children],
        },
        "producer": {
            "status": "dd" if dd_text else ("derived" if name else "none"),
            "dd_paths": list(item.get("source_paths") or ()),
            "derived": [],
            "statement": "a producer answer was computed from the item",
        },
    }


def record_stage_receipt(
    stage: str,
    name: str,
    context: Mapping[str, Any],
    *,
    prompt_text: str = "",
    sink: Path | None = None,
) -> ContextReceipt:
    """Record the receipt for a stage that has just rendered its prompt.

    This is the production entry point: a stage calls it with the context it
    sent and the prompt it rendered, and the receipt records what arrived. It
    is observation only and never touches a model. The receipt is appended to
    the sink :func:`receipt_sink` returns — by default the CLI's own log
    directory — so calling this is what makes a stage's record exist; it is
    not a no-op when nothing configured a sink.
    """
    content = content_from_prompt_context(context, name=name)
    receipt = receipt_from_content(
        stage, str(name or content["target"]["id"]), content, prompt_text=prompt_text
    )
    record_receipt(receipt, sink=sink)
    return receipt
