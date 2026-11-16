"""Measure a stored controls feed against a read-only archive catalogue.

All input and output paths are explicit. This script performs no network or
graph operations. Its JSON output is private: it includes source identities.
"""

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


class MeasurementError(ValueError):
    """The input cannot support a coverage measurement."""


def parse_entries(body):
    """Accept the installed client's list and single-list wrapper shapes."""
    if isinstance(body, dict):
        lists = [v for v in body.values() if isinstance(v, list)]
        if len(lists) != 1:
            raise MeasurementError(
                "response wrapper must contain exactly one entry list"
            )
        body = lists[0]
    if not isinstance(body, list):
        raise MeasurementError("response must be an entry list or single-list wrapper")
    if not body:
        raise MeasurementError("response contains no entries")
    entries = []
    for index, row in enumerate(body):
        if not isinstance(row, dict):
            raise MeasurementError(f"entry {index} must be an object")
        name = row.get("variable") or row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MeasurementError(f"entry {index} has no non-empty string name")
        fields = {
            "description": row.get("description", ""),
            "unit": row.get("unit", row.get("units", "")),
            "type": row.get("data_type", row.get("data type", row.get("type", ""))),
        }
        for key, value in fields.items():
            if value is not None and not isinstance(value, str):
                raise MeasurementError(f"entry {index} has non-string {key}")
        entries.append(
            {"name": name, **{k: (v or "").strip() for k, v in fields.items()}}
        )
    return entries


NORMALISATIONS = {
    "exact": lambda s: s,
    "outer_whitespace": lambda s: s.strip(),
    "casefold": lambda s: s.casefold(),
    "colon_to_hyphen": lambda s: s.replace(":", "-"),
    "colon_to_dot": lambda s: s.replace(":", "."),
    "separator_fold_casefold": lambda s: re.sub(r"[-:_.]+", "-", s.strip().casefold()),
}


def plant_system(name):
    return name.split(":", 1)[0].split("-", 1)[0]


def in_cohort(name, token):
    """Match an entire component field, including diagnostic namespaces."""
    return token.casefold() in name.split(":", 1)[0].casefold().split("-")


def feed_quality(entries):
    descriptions = Counter(r["description"] for r in entries)
    lengths = sorted(len(r["description"]) for r in entries)
    top = Counter({k: n for k, n in descriptions.items() if k}).most_common(10)
    names = {r["name"] for r in entries}
    return {
        "entries": len(entries),
        "unique_names": len(names),
        "duplicate_name_entries": len(entries) - len(names),
        "nonempty": {
            k: sum(bool(r[k]) for r in entries) for k in ("description", "unit", "type")
        },
        "distinct_descriptions_including_empty": len(descriptions),
        "distinct_nonempty_descriptions": len(descriptions) - ("" in descriptions),
        "ten_most_common": top,
        "ten_most_common_count": sum(n for _, n in top),
        "ten_most_common_share": sum(n for _, n in top) / len(entries),
        "description_length_quantiles": {
            str(p): lengths[max(0, math.ceil(p / 100 * len(lengths)) - 1)]
            for p in (0, 10, 25, 50, 75, 90, 95, 100)
        },
        "description_length_bins": {
            label: sum(lo <= len(r["description"]) <= hi for r in entries)
            for label, lo, hi in (
                ("empty", 0, 0),
                ("1–9", 1, 9),
                ("10–19", 10, 19),
                ("20–39", 20, 39),
                ("40–79", 40, 79),
                ("80+", 80, math.inf),
            )
        },
        "description_equals_name_or_suffix": sum(
            r["description"] in (r["name"], r["name"].split(":")[-1]) for r in entries
        ),
    }


def compare_catalogue(entries, catalogue, control_name, cohort_token):
    """Stream distinct archive names; count feed matches and archive coverage."""
    feed_names = {r["name"] for r in entries}
    indexes = {}
    for label, normalise in NORMALISATIONS.items():
        index = defaultdict(set)
        for name in feed_names:
            index[normalise(name)].add(name)
        indexes[label] = index
    matched_feed = {label: set() for label in indexes}
    matched_archive_keys = {label: Counter() for label in indexes}
    archive_counts = Counter()
    archive_matches = {label: Counter() for label in indexes}
    cohort_total = 0
    cohort_matches = Counter()
    cohort_feed = {name for name in feed_names if in_cohort(name, cohort_token)}
    uri = catalogue.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        if not connection.execute(
            "SELECT 1 FROM census WHERE name = ? LIMIT 1", (control_name,)
        ).fetchone():
            raise MeasurementError("known archive name positive control was not found")
        for (name,) in connection.execute("SELECT DISTINCT name FROM census"):
            if not isinstance(name, str) or not name:
                raise MeasurementError("archive contains an invalid name")
            system = plant_system(name)
            archive_counts[system] += 1
            cold = in_cohort(name, cohort_token)
            cohort_total += cold
            for label, normalise in NORMALISATIONS.items():
                key = normalise(name)
                if key in indexes[label]:
                    archive_matches[label][system] += 1
                    matched_feed[label].update(indexes[label][key])
                    matched_archive_keys[label][key] += 1
                    if cold:
                        cohort_matches[label] += 1
    total = sum(archive_counts.values())
    if not total:
        raise MeasurementError("archive catalogue contains no names")
    systems = []
    for number, system in enumerate(
        sorted(archive_counts, key=lambda s: hashlib.sha256(s.encode()).hexdigest()), 1
    ):
        systems.append(
            {
                "private_system": system,
                "public_label": f"Plant system {number:02d}",
                "catalogue": archive_counts[system],
                "feed": sum(plant_system(n) == system for n in feed_names),
                "matches": {k: v[system] for k, v in archive_matches.items()},
            }
        )
    return {
        "catalogue_names": total,
        "positive_control_found": True,
        "normalisations": {
            label: {
                "matched_feed_names": len(matched_feed[label]),
                "covered_archive_names": sum(archive_matches[label].values()),
                "archive_coverage": sum(archive_matches[label].values()) / total,
                "feed_collision_keys": sum(len(v) > 1 for v in indexes[label].values()),
                "matched_archive_collision_keys": sum(
                    n > 1 for n in matched_archive_keys[label].values()
                ),
            }
            for label in indexes
        },
        "cohort": {
            "private_token": cohort_token,
            "feed_names": len(cohort_feed),
            "catalogue_names": cohort_total,
            "covered_archive_names": dict(cohort_matches),
        },
        "systems": sorted(systems, key=lambda row: -row["catalogue"]),
    }


def measure(response, catalogue, settings):
    raw = response.read_bytes()
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise MeasurementError("stored response is not valid JSON") from exc
    entries = parse_entries(body)
    sample = sorted(
        entries,
        key=lambda r: (hashlib.sha256(r["name"].encode()).hexdigest(), r["name"]),
    )[:30]
    rows = (
        body
        if isinstance(body, list)
        else next(v for v in body.values() if isinstance(v, list))
    )
    return {
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "quality": feed_quality(entries),
        "coverage": compare_catalogue(
            entries, catalogue, settings["control_name"], settings["cohort_token"]
        ),
        "hand_sample": sample,
        "timestamp_fields": {
            key: dict(Counter(r.get(key) for r in rows))
            for key in ("insertion_time", "update_time")
        },
        "response_wrapper_metadata": (
            {k: v for k, v in body.items() if not isinstance(v, list)}
            if isinstance(body, dict)
            else {}
        ),
    }


def plot_coverage(result, path):
    """Plot all plant systems, using private-to-public labels from the receipt."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from imas_codex.plotting import save_figure
    from imas_codex.plotting.style import BASE_STYLE, CANVAS, SUBJECT

    systems = result["coverage"]["systems"]
    with plt.style.context(BASE_STYLE):
        figure, axis = plt.subplots(
            figsize=(CANVAS[0], max(CANVAS[1], len(systems) * 0.31))
        )
        figure.subplots_adjust(left=0.23, right=0.95, bottom=0.12, top=0.96)
        ys = list(range(len(systems)))
        values = [100 * row["matches"]["exact"] / row["catalogue"] for row in systems]
        axis.scatter(values, ys, color=SUBJECT, s=45, zorder=3)
        axis.set_yticks(ys, [row["public_label"] for row in systems])
        axis.invert_yaxis()
        axis.set_xlabel("Archive names covered by exact feed match [%]")
        axis.set_xlim(-0.3, max(1, max(values)) * 1.08)
        axis.xaxis.set_major_locator(MaxNLocator(6))
        axis.spines[["left", "top", "right"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        save_figure(figure, path)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.response, args.catalogue, json.loads(args.settings.read_text())
        )
    except MeasurementError as exc:
        parser.exit(1, f"measurement failed: {exc}\n")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.chmod(0o600)
    if args.figure:
        plot_coverage(result, args.figure)
    print(
        json.dumps(
            {
                "entries": result["quality"]["entries"],
                "catalogue_names": result["coverage"]["catalogue_names"],
                "exact_matches": result["coverage"]["normalisations"]["exact"][
                    "covered_archive_names"
                ],
                "positive_control_found": result["coverage"]["positive_control_found"],
            }
        )
    )


if __name__ == "__main__":
    main()
