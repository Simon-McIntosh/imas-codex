"""Every configured model seat names a current, cataloged and priced model.

Two properties are asserted here, both read from the configuration rather than
from a copy of it:

* each seat names the newest release of the model line it intends, established
  from the provider catalogs recorded in ``data/model-catalog-2026-11-15.json``
  — the OpenRouter model list, the local lane's served models, the decisions
  endpoint's served releases and the embedding hub listing; and
* every OpenRouter identifier a seat or panel names carries a cataloged price,
  so a seat cannot start a run it is unable to price.

The catalog is a snapshot taken on the date in its filename, not a live query:
a provider releasing a model does not edit this repository, so the check
measures the seats against the catalog recorded on that date. Refreshing it is
deliberate — read the four catalogs again, rewrite the snapshot with its
sources and read date, and re-run this file; a seat that has fallen behind the
refreshed snapshot fails here with the newest release named.

A seat may stay off its line's newest release only through an entry in the
snapshot's ``exceptions`` map, which carries the reason it stays, so a stale
seat is either repaired or explained rather than silently tolerated.

The escalation seat gets its own test because it is the one seat whose model is
chosen by the refine rotation at chain cap rather than by the caller, and
because its reason for existing is that it stays a different vendor from the
refine and compose seats.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from iter_codex.settings import (
    _MODEL_DEFAULTS,
    _SN_BENCHMARK_DEFAULTS,
    _SN_REVIEW_DEFAULTS,
    _get_section,
    get_model,
    get_sn_benchmark_candidate_models,
    get_sn_benchmark_compose_models,
    get_sn_benchmark_reviewer_model,
    get_sn_benchmark_reviewer_models,
    get_sn_review_docs_models,
    get_sn_review_names_models,
    get_sn_review_profile_models,
)
from iter_codex.standard_names.defaults import DEFAULT_REFINE_ROTATIONS
from iter_codex.standard_names.worker_tasks.worker_refine_names import (
    refine_model_for_attempt,
)
from tests.standard_names.test_seat_exposure_calibration import _assert_proven_pricing

# Every single-model seat under [tool.iter-codex.*].
SINGLE_SEATS = [
    "sn-docs",
    "sn-refine",
    "sn-escalation",
    "sn-parent-enrich",
    "sn-classifier",
    "sn-domain-tree",
    "sn-prose-adjudicator",
    "sn-release-notes",
    "dd-enrichment",
    "reasoning",
    "cluster-labels",
    "ids-mapping",
    "discovery-triage",
    "discovery-score",
    "discovery-describe",
    "discovery-vision",
    "discovery-relevance",
    "mapping-candidates",
    "sn-judge",
    "sn-compose",
]

REVIEW_PROFILES = ("default", "local-only", "quality-cost-balanced", "opus-only")

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_CATALOG = Path(__file__).with_name("data") / "model-catalog-2026-11-15.json"
_VERSION_TOKEN = re.compile(r"^[a-z]*\d+(?:\.\d+)*$", re.IGNORECASE)
_MODEL_KEY = re.compile(r"(^|-)models?$")
_DATED_BUILD = re.compile(r"^(?P<release>.+)-\d{4}$")


def _catalog() -> dict:
    return json.loads(_CATALOG.read_text())


def _declared_models() -> dict[str, str]:
    """Every model identifier a ``pyproject.toml`` section declares, by owner.

    The owner names the section and query path it was found at
    (``sn-docs.model``, ``sn-review.names.models[0]``,
    ``sn-fanout.proposer-model``), so a seat added to the configuration is
    covered by these checks without an edit here.
    """
    config = tomllib.loads(_PYPROJECT.read_text())["tool"]["iter-codex"]
    found: dict[str, str] = {}

    def walk(node: object, path: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if not _MODEL_KEY.search(key):
                if isinstance(value, dict):
                    walk(value, child)
                continue
            if isinstance(value, str):
                found[child] = value
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, str):
                        found[f"{child}[{index}]"] = item

    walk(config, "")
    return found


def _line(model_id: str) -> str:
    """The model line an identifier belongs to, with its version removed.

    Version tokens are dropped and tier words kept, so ``claude-sonnet-5`` and
    ``claude-sonnet-5.5`` share a line while ``claude-opus-5.5`` does not, and
    a release date decides between ``grok-4.20`` and ``grok-4.7`` rather than
    string order. A provider generation number counts as part of the version:
    ``deepseek-v4-flash`` and ``deepseek-v4.1-flash`` are versions of one flash
    line.
    """
    vendor, _, family = model_id.partition("/")
    words = [word for word in family.split("-") if not _VERSION_TOKEN.match(word)]
    return f"{vendor}/" + "-".join(words)


def _newest_release(releases: dict[str, str], model_id: str) -> str | None:
    """The newest release of *model_id*'s line, ordered by release date.

    A dated build such as ``deepseek-v4.1-flash-1017`` is one provider's
    snapshot of a release whose undated identifier is listed beside it; the
    undated identifier is the version's own name, so the dated listing is
    dropped and cannot outrank the alias it snapshots.
    """
    line = _line(model_id)
    members = {mid: day for mid, day in releases.items() if _line(mid) == line}
    for mid in list(members):
        dated = _DATED_BUILD.match(mid)
        if dated and dated["release"] in members:
            del members[mid]
    if not members:
        return None
    return max(members, key=lambda mid: members[mid])


def _panel_models() -> list[str]:
    models = list(get_sn_review_names_models())
    models += list(get_sn_review_docs_models())
    for profile in REVIEW_PROFILES:
        models += list(get_sn_review_profile_models(profile))
    models += list(get_sn_benchmark_compose_models())
    models += list(get_sn_benchmark_candidate_models())
    models += list(get_sn_benchmark_reviewer_models())
    models.append(get_sn_benchmark_reviewer_model())
    return models


def _assert_priced(model: str, owner: str) -> None:
    if not model.startswith("openrouter/"):
        # A local or decisions seat is priced by its own lane: the router's
        # endpoint class, or the decisions response's own usage cost.
        return
    pricing = _assert_proven_pricing(owner, model)
    listed = _catalog()["openrouter"]["models"][model]["pricing"]
    for key in ("prompt", "completion"):
        assert pricing[key] == pytest.approx(float(listed[key]) * 1_000_000), (
            f"{owner} prices {model!r} differently from its serving catalogue"
        )
    cache_read = pricing["cache_read"]
    assert isinstance(cache_read, (int, float)), (
        f"{owner} prices {model!r} with cache_read={cache_read!r}"
    )
    # A route may publish no cache-write rate, in which case the admission
    # estimate prices those tokens at the base prompt rate rather than at an
    # invented premium.
    cache_write = pricing["cache_write"]
    assert cache_write is None or isinstance(cache_write, (int, float)), (
        f"{owner} prices {model!r} with cache_write={cache_write!r}"
    )


@pytest.mark.parametrize("seat", SINGLE_SEATS)
def test_every_seat_resolves_and_is_priced(seat: str) -> None:
    model = get_model(seat)
    assert model, f"seat {seat!r} resolved to an empty model"
    _assert_priced(model, f"seat {seat!r}")


@pytest.mark.parametrize("model", sorted(set(_panel_models())))
def test_panel_identifiers_are_priced(model: str) -> None:
    _assert_priced(model, "a review panel or benchmark slate")


def _python_default_identifiers() -> list[tuple[str, str]]:
    """Every OpenRouter identifier a Python default names, with its owner.

    Read from the default mappings themselves rather than through the
    accessors: the accessors return the project configuration whenever it
    exists, so a default that names an unpriceable route is never exercised by
    them.
    """
    rows = [
        (f"_MODEL_DEFAULTS[{section!r}]", model)
        for section, model in _MODEL_DEFAULTS.items()
        if isinstance(model, str)
    ]
    for key, value in _SN_BENCHMARK_DEFAULTS.items():
        values = value if isinstance(value, list) else [value]
        rows += [(f"_SN_BENCHMARK_DEFAULTS[{key!r}]", model) for model in values]
    return [(owner, model) for owner, model in rows if model.startswith("openrouter/")]


@pytest.mark.parametrize(
    "owner,model", _python_default_identifiers(), ids=lambda value: str(value)
)
def test_python_default_identifiers_are_priced(owner: str, model: str) -> None:
    _assert_priced(model, owner)


_DECLARED = sorted(_declared_models().items(), key=lambda row: row[0])


def test_the_declared_seat_list_is_not_vacuous() -> None:
    """The scan finds the seats it claims to check, so a green run means something."""
    declared = _declared_models()
    assert len(declared) >= 20, (
        f"the pyproject.toml scan found only {len(declared)} declared models; "
        "the key pattern stopped matching and the parametrised checks below "
        "would pass without checking a seat"
    )
    for known in (
        "sn-compose.model",
        "sn-review.names.models[0]",
        "sn-fanout.proposer-model",
    ):
        assert known in declared, (
            f"the scan missed {known!r}, a seat this project declares; a scan "
            "that misses a seat reports a green result it never earned"
        )


@pytest.mark.parametrize("owner,model", _DECLARED, ids=[row[0] for row in _DECLARED])
def test_every_declared_seat_names_a_cataloged_model(owner: str, model: str) -> None:
    """Each seat's model is listed by the lane its prefix routes to."""
    catalog = _catalog()
    if model.startswith("openrouter/"):
        listed = catalog["openrouter"]["releases"]
        where = catalog["openrouter"]["source"]
    elif model.startswith("local/"):
        listed = catalog["local"]["served"]
        where = catalog["local"]["source"]
        model = model.removeprefix("local/")
    elif model.startswith("typesafe/"):
        listed = catalog["decisions"]["served"]
        where = catalog["decisions"]["source"]
    else:
        listed = catalog["embedding"]["served"]
        where = catalog["embedding"]["source"]
    assert model in listed, (
        f"{owner} names {model!r}, which {where} does not list; read the "
        "catalog again and record the new release in the snapshot"
    )


@pytest.mark.parametrize("owner,model", _DECLARED, ids=[row[0] for row in _DECLARED])
def test_every_seat_names_the_newest_release_of_its_line(
    owner: str, model: str
) -> None:
    """A seat is on its line's newest release, or records why it stays."""
    catalog = _catalog()
    if model.startswith("openrouter/"):
        successor = catalog["openrouter"]["tier_successors"].get(model)
        assert successor is None, (
            f"{owner} names {model!r}, but its provider's newest same-tier "
            f"line is {successor!r}; see the snapshot's tier_judgements"
        )
        newest = _newest_release(catalog["openrouter"]["releases"], model)
    elif model.startswith("typesafe/"):
        served = catalog["decisions"]["served"]

        def release_number(candidate: str) -> tuple[int, ...]:
            return tuple(int(part) for part in candidate.rsplit("-", 1)[-1].split("."))

        newest = max(served, key=release_number)
    elif model.startswith("local/"):
        # The lane serves one model per served name; its catalog is the lane's
        # own list, so any served name is the lane's current model and a name
        # the lane has retired is the stale case covered above.
        return
    else:
        # Embedding sizes are the releases of one line; no newer text-embedding
        # release exists in the hub listing, so the listing is the authority.
        return

    if owner in catalog["exceptions"]:
        assert catalog["exceptions"][owner], (
            f"{owner} is excepted from its line's newest release {newest!r} "
            "but records no reason it stays"
        )
        return
    assert newest is not None, f"{owner} names {model!r}, which has no cataloged line"
    assert model == newest, (
        f"{owner} names {model!r}, but its line's newest release is {newest!r} "
        f"({catalog['openrouter']['releases'].get(newest, 'served today')}). "
        "Name the newest release, record why the seat stays in the catalog "
        f"snapshot's exceptions map keyed by {owner!r}, or — when the provider "
        "released this model after the snapshot was taken — refresh the "
        "snapshot and re-run this file"
    )


@pytest.mark.parametrize("model", sorted(set(_panel_models())))
def test_panel_models_are_on_their_lines_newest_release(model: str) -> None:
    if not model.startswith("openrouter/"):
        return
    newest = _newest_release(_catalog()["openrouter"]["releases"], model)
    assert newest is not None, f"a panel names {model!r}, which has no cataloged line"
    assert model == newest, (
        f"a panel names {model!r}, but its line's newest release is {newest!r}"
    )


def test_no_seat_names_a_release_the_decisions_lane_refuses() -> None:
    """The decisions endpoint's refusals are the negative arm of its listing."""
    catalog = _catalog()
    refused = set(catalog["decisions"]["refused"])
    served = set(catalog["decisions"]["served"])
    assert refused and served, "the decisions ladder recorded nothing to compare"
    assert not refused & served
    named = {model for owner, model in _DECLARED if model.startswith("typesafe/")}
    assert named and not named & refused, (
        f"a seat names a refused decisions release: {sorted(named & refused)}"
    )


@pytest.mark.parametrize("owner,model", _DECLARED, ids=[row[0] for row in _DECLARED])
def test_every_declared_seat_is_generally_available(owner: str, model: str) -> None:
    assert not re.search(r"(?:^|[-/])(preview|beta|experimental|exp)(?:$|-)", model), (
        f"{owner} names a non-production model: {model!r}"
    )
    if model.startswith("openrouter/"):
        expiration = _catalog()["openrouter"]["models"][model]["expiration_date"]
        assert expiration is None or expiration > _catalog()["read_at"], (
            f"{owner} names expired model {model!r}: {expiration}"
        )


@pytest.mark.parametrize(
    "section", sorted(section for section in _MODEL_DEFAULTS if _get_section(section))
)
def test_model_fallback_equals_configured_seat(section: str) -> None:
    assert _MODEL_DEFAULTS[section] == _get_section(section)["model"]


@pytest.mark.parametrize("axis", ["names", "docs"])
def test_review_fallback_equals_configured_seats(axis: str) -> None:
    assert (
        _SN_REVIEW_DEFAULTS[f"{axis}-models"]
        == _get_section("sn-review")[axis]["models"]
    )


@pytest.mark.parametrize("key", sorted(_SN_BENCHMARK_DEFAULTS))
def test_benchmark_fallback_equals_configured_seats(key: str) -> None:
    assert _SN_BENCHMARK_DEFAULTS[key] == _get_section("sn-benchmark")[key]


def test_escalation_seat_drives_the_final_refine_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refine worker's last rotation runs the escalation seat's model.

    The seat is pointed at another priced model through its environment
    override, so the rotation has to resolve the seat at call time. A rotation
    that ignores the seat, reading either a built-in constant or the refine
    tier, resolves a model the seat does not name and fails here.
    """
    escalated_model = "openrouter/x-ai/grok-4.7"
    monkeypatch.setenv("ITER_CODEX_SN_ESCALATION_MODEL", escalated_model)

    seat = _get_section("sn-escalation")
    cap = DEFAULT_REFINE_ROTATIONS
    final_model, final_escalates = refine_model_for_attempt(cap, cap)
    assert final_escalates is True
    assert final_model == escalated_model, (
        "the final rotation resolved a model the escalation seat does not name"
    )
    assert final_model == get_model("sn-escalation")
    assert seat["model"] != escalated_model, (
        "the override did not move the seat, so this run proves nothing"
    )

    earlier_model, earlier_escalates = refine_model_for_attempt(1, cap)
    assert earlier_escalates is False
    assert earlier_model == get_model("sn-refine")


def test_escalation_stays_vendor_diverse_from_refine_and_compose() -> None:
    def vendor(model: str) -> str:
        parts = model.split("/")
        return parts[1] if model.startswith(("openrouter/", "local/")) else parts[0]

    escalation = vendor(get_model("sn-escalation"))
    assert escalation != vendor(get_model("sn-refine"))
    assert escalation != vendor(get_model("sn-compose"))
