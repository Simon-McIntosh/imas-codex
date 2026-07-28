"""Version arithmetic for the ISNC catalog release — including batch RCs.

A release cut against a review batch carries the batch identity in the version
itself as semver **build metadata** (``v0.2.0rc65+<label>``). These tests pin:

* the sanitizer that turns a manifest name into a legal build-metadata string;
* that the tag grammar, state detection, and RC arithmetic all see a batch RC
  (before this, ``v0.2.0rc65+label`` did not match the tag grammar at all, so
  state detection skipped it and the next release silently reused rc65);
* that build metadata never affects precedence and never lands on a stable tag.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from imas_codex.standard_names.catalog_release import (
    _format_tag,
    _get_semver_tags,
    _parse_build,
    _parse_version,
    batch_build_metadata,
    compute_next_version,
    detect_state,
    sanitize_build_metadata,
)


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def tagged_repo(tmp_path):
    """A repo with one commit; each test adds whatever tags it needs."""
    work = tmp_path / "isnc"
    work.mkdir()
    _git("init", "-q", "-b", "main", cwd=work)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    (work / "README.md").write_text("x\n")
    _git("add", "README.md", cwd=work)
    _git("commit", "-qm", "init", cwd=work)
    return work


# ---------------------------------------------------------------------------
# The sanitizer
# ---------------------------------------------------------------------------


class TestSanitizeBuildMetadata:
    def test_underscores_become_hyphens(self):
        assert sanitize_build_metadata("west_task_2e") == "west-task-2e"

    def test_already_legal_label_is_preserved(self):
        assert sanitize_build_metadata("west-task-2e") == "west-task-2e"

    def test_dots_are_legal_identifier_separators(self):
        assert sanitize_build_metadata("west.task.2e") == "west.task.2e"

    def test_case_is_folded(self):
        assert sanitize_build_metadata("WEST_Task_2E") == "west-task-2e"

    def test_illegal_characters_collapse_to_a_single_hyphen(self):
        assert sanitize_build_metadata("west/task (2e)!") == "west-task-2e"

    def test_leading_and_trailing_separators_are_stripped(self):
        assert sanitize_build_metadata("__west__") == "west"
        assert sanitize_build_metadata("..west..") == "west"

    def test_empty_input_is_an_error(self):
        with pytest.raises(ValueError):
            sanitize_build_metadata("")

    def test_all_illegal_input_is_an_error(self):
        with pytest.raises(ValueError):
            sanitize_build_metadata("///")

    def test_output_is_legal_semver_build_metadata(self):
        label = sanitize_build_metadata("WEST task_2e/v3 (final)")
        assert all(c.isalnum() or c in "-." for c in label)
        assert not label.startswith((".", "-"))
        assert not label.endswith((".", "-"))


class TestBatchBuildMetadata:
    def test_derived_from_the_manifest_name_field(self, tmp_path):
        p = tmp_path / "somefile.yaml"
        p.write_text("kind: sn_sources\nname: demo_batch\n", encoding="utf-8")
        assert batch_build_metadata(p) == "demo-batch"

    def test_falls_back_to_the_filename_stem(self, tmp_path):
        p = tmp_path / "fallback_batch.yaml"
        p.write_text("kind: sn_sources\n", encoding="utf-8")
        assert batch_build_metadata(p) == "fallback-batch"

    def test_frozen_artifact_double_suffix_stem_is_usable(self, tmp_path):
        p = tmp_path / "v0.2.0rc65+demo.sn_names.yaml"
        p.write_text("kind: sn_names\nname: review-v0-2-0rc65\n", encoding="utf-8")
        assert batch_build_metadata(p) == "review-v0-2-0rc65"

    def test_unusable_name_and_stem_raise(self, tmp_path):
        p = tmp_path / "---.yaml"
        p.write_text("kind: sn_sources\nname: '///'\n", encoding="utf-8")
        with pytest.raises(ValueError):
            batch_build_metadata(p)

    def test_a_committed_manifest_yields_its_declared_name(self):
        manifests = (
            Path(__file__).resolve().parents[2]
            / "imas_codex"
            / "standard_names"
            / "manifests"
        )
        checked = 0
        for candidate in sorted(manifests.glob("*.yaml")):
            doc = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or not doc.get("name"):
                continue
            assert batch_build_metadata(candidate) == sanitize_build_metadata(
                doc["name"]
            )
            checked += 1
        assert checked, "no committed manifest declares a name"


# ---------------------------------------------------------------------------
# Tag grammar
# ---------------------------------------------------------------------------


class TestTagGrammar:
    def test_plain_and_rc_tags_still_parse(self):
        assert _parse_version("v1.2.3") == (1, 2, 3, None)
        assert _parse_version("v0.2.0rc65") == (0, 2, 0, 65)

    def test_batch_rc_tag_parses(self):
        assert _parse_version("v0.2.0rc65+west-task-2e") == (0, 2, 0, 65)
        assert _parse_build("v0.2.0rc65+west-task-2e") == "west-task-2e"

    def test_no_build_metadata_reads_as_none(self):
        assert _parse_build("v0.2.0rc65") is None

    def test_malformed_tags_are_rejected(self):
        for bad in (
            "0.2.0rc1",
            "v0.2rc1",
            "v0.2.0-rc1",
            "v0.2.0rc1+",
            "v0.2.0rc1+a b",
            "v0.2.0rc1+a_b",
            "v0.3.0-rc1-w40-corpus",
        ):
            with pytest.raises(ValueError):
                _parse_version(bad)

    def test_format_round_trips_build_metadata(self):
        tag = _format_tag(0, 2, 0, 65, build="west-task-2e")
        assert tag == "v0.2.0rc65+west-task-2e"
        assert _parse_version(tag) == (0, 2, 0, 65)
        assert _parse_build(tag) == "west-task-2e"

    def test_a_stable_tag_can_never_carry_build_metadata(self):
        with pytest.raises(ValueError):
            _format_tag(1, 0, 0, None, build="west-task-2e")


# ---------------------------------------------------------------------------
# State detection + precedence
# ---------------------------------------------------------------------------


class TestDetectState:
    def test_batch_rc_is_the_detected_latest(self, tagged_repo):
        _git("tag", "v0.2.0rc64", cwd=tagged_repo)
        _git("tag", "v0.2.0rc65+west-task-2e", cwd=tagged_repo)
        info = detect_state(tagged_repo)
        assert info["state"] == "rc"
        assert info["tag"] == "v0.2.0rc65+west-task-2e"
        assert (info["major"], info["minor"], info["patch"], info["rc"]) == (
            0,
            2,
            0,
            65,
        )
        assert info["build"] == "west-task-2e"

    def test_plain_rc_reports_no_build_metadata(self, tagged_repo):
        _git("tag", "v0.2.0rc65", cwd=tagged_repo)
        assert detect_state(tagged_repo)["build"] is None

    def test_build_metadata_does_not_change_precedence(self, tagged_repo):
        _git("tag", "v0.2.0rc65+aaa", cwd=tagged_repo)
        _git("tag", "v0.2.0rc66", cwd=tagged_repo)
        assert _get_semver_tags(cwd=tagged_repo)[0] == "v0.2.0rc66"

    def test_rc_ordering_is_numeric_not_lexical(self, tagged_repo):
        _git("tag", "v0.2.0rc9", cwd=tagged_repo)
        _git("tag", "v0.2.0rc65+aaa", cwd=tagged_repo)
        assert _get_semver_tags(cwd=tagged_repo)[0] == "v0.2.0rc65+aaa"

    def test_labelled_and_bare_rc_share_one_precedence_rank(self, tagged_repo):
        _git("tag", "v0.2.0rc65+aaa", cwd=tagged_repo)
        _git("tag", "v0.2.0rc65", cwd=tagged_repo)
        _git("tag", "v0.2.0rc64", cwd=tagged_repo)
        tags = _get_semver_tags(cwd=tagged_repo)
        # Equal precedence: both rc65 forms rank above rc64, in either order.
        assert set(tags[:2]) == {"v0.2.0rc65+aaa", "v0.2.0rc65"}
        assert tags[2] == "v0.2.0rc64"
        # Whichever is picked, the arithmetic is identical.
        assert detect_state(tagged_repo)["rc"] == 65

    def test_legacy_suffixed_tags_are_still_ignored(self, tagged_repo):
        _git("tag", "v0.3.0-rc1-w40-corpus", cwd=tagged_repo)
        _git("tag", "v0.2.0rc65+aaa", cwd=tagged_repo)
        assert _get_semver_tags(cwd=tagged_repo) == ["v0.2.0rc65+aaa"]


# ---------------------------------------------------------------------------
# RC arithmetic
# ---------------------------------------------------------------------------


class TestComputeNextVersion:
    def test_next_rc_continues_the_counter_past_a_batch_rc(self, tagged_repo):
        """A batch RC must advance the counter, not be skipped over.

        If the batch tag is invisible to the tag grammar, state detection falls
        back to the previous bare RC and the next release re-mints an RC number
        that already has a tag.
        """
        _git("tag", "v0.2.0rc64", cwd=tagged_repo)
        _git("tag", "v0.2.0rc65+west-task-2e", cwd=tagged_repo)
        tag, version = compute_next_version(tagged_repo, None)
        assert tag == "v0.2.0rc66"
        assert version == "0.2.0rc66"

    def test_next_batch_rc_carries_the_new_label_not_the_old(self, tagged_repo):
        _git("tag", "v0.2.0rc65+west-task-2e", cwd=tagged_repo)
        tag, version = compute_next_version(tagged_repo, None, build="other-batch")
        assert tag == "v0.2.0rc66+other-batch"
        assert version == "0.2.0rc66+other-batch"

    def test_final_from_a_batch_rc_drops_the_metadata(self, tagged_repo):
        _git("tag", "v0.2.0rc65+west-task-2e", cwd=tagged_repo)
        tag, _ = compute_next_version(tagged_repo, None, final=True)
        assert tag == "v0.2.0"

    def test_final_refuses_build_metadata(self, tagged_repo):
        _git("tag", "v0.2.0rc65", cwd=tagged_repo)
        with pytest.raises(ValueError):
            compute_next_version(tagged_repo, None, final=True, build="west-task-2e")

    def test_bump_from_a_batch_rc_restarts_at_rc1(self, tagged_repo):
        _git("tag", "v0.2.0", cwd=tagged_repo)
        _git("tag", "v0.2.0rc65+west-task-2e", cwd=tagged_repo)
        tag, _ = compute_next_version(tagged_repo, "minor", build="west-task-2e")
        assert tag == "v0.3.0rc1+west-task-2e"

    def test_first_release_can_carry_a_label(self, tagged_repo):
        tag, _ = compute_next_version(tagged_repo, "minor", build="west-task-2e")
        assert tag == "v0.1.0rc1+west-task-2e"

    def test_non_batch_release_is_unchanged(self, tagged_repo):
        """Regression guard: no label anywhere when no batch is given."""
        _git("tag", "v0.2.0rc65", cwd=tagged_repo)
        tag, version = compute_next_version(tagged_repo, None)
        assert tag == "v0.2.0rc66"
        assert "+" not in tag and "+" not in version
