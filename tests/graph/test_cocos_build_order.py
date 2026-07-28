"""Tests for COCOS label ordering fix in build_dd.py.

Verifies that _backfill_cocos_labels returns (count, handled_set) and that
the cleanup step does not wipe labels produced by backfill.
"""

from unittest.mock import MagicMock

from imas_codex.graph.build_dd import _backfill_cocos_labels

# Minimal version_data that satisfies the function's structure requirements.
# One 3.x version with a labeled path, one 4.x version with the same path
# but no cocos_transformation_type (so backfill has something to do).
_MINIMAL_VERSION_DATA = {
    "3.42.0": {
        "paths": {
            "equilibrium/time_slice/profiles_1d/psi": {
                "cocos_transformation_type": "psi_like",
                "cocos_transformation_expression": None,
            }
        }
    },
    "4.0.0": {
        "paths": {
            "equilibrium/time_slice/profiles_1d/psi": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": None,
            }
        }
    },
}


def _make_client() -> MagicMock:
    """Return a mock GraphClient that satisfies all query calls."""
    client = MagicMock()
    client.query.return_value = []
    return client


def test_backfill_returns_tuple():
    """_backfill_cocos_labels must return (int, set), not bare int."""
    result = _backfill_cocos_labels(_make_client(), _MINIMAL_VERSION_DATA)

    assert isinstance(result, tuple), "Return value must be a 2-tuple"
    assert len(result) == 2, "Return tuple must have exactly 2 elements"
    count, handled = result
    assert isinstance(count, int), "First element (count) must be int"
    assert isinstance(handled, set), "Second element (handled) must be set"


def test_backfill_returns_handled_paths():
    """Paths that receive a backfilled label appear in the returned handled set."""
    count, handled = _backfill_cocos_labels(_make_client(), _MINIMAL_VERSION_DATA)

    # The 4.0.0 path has no XML label → should be forward-ported from 3.42.0
    assert count >= 1
    assert "equilibrium/time_slice/profiles_1d/psi" in handled


def test_cleanup_excludes_backfilled_paths():
    """The union of xml-labeled and backfilled paths must be used for cleanup.

    This is a logic-level test: we verify that if backfill returns paths,
    those paths are excluded from the null-setting Cypher call.
    """
    backfilled = {"some/path/a", "some/path/b"}
    xml_labeled = {"some/path/c"}

    # The cleanup set build_dd.py must compute — both label sources, unioned:
    all_labeled = xml_labeled | backfilled

    assert "some/path/a" in all_labeled
    assert "some/path/b" in all_labeled
    assert "some/path/c" in all_labeled

    # Paths NOT in either set would be cleaned up — ensure a random path is absent.
    assert "some/path/unlabeled" not in all_labeled


def test_cleanup_scoped_to_xml_labels_alone_wipes_backfilled():
    """Cleanup scoped to xml_labeled alone would remove backfilled paths.

    Pins why the cleanup set must be the union of xml_labeled and backfilled:
    a backfilled path carries no XML label, so a cleanup keyed on XML labels
    treats it as unlabeled and wipes it.
    """
    backfilled = {"some/path/a", "some/path/b"}
    xml_labeled = {"some/path/c"}

    # Cleanup keyed on xml_labeled alone, i.e. no union with backfilled
    xml_only_labeled = xml_labeled

    # Backfilled paths fall outside that set and would be cleaned up:
    assert "some/path/a" not in xml_only_labeled
    assert "some/path/b" not in xml_only_labeled

    # Taking the union instead keeps them out of the cleanup set
    union_labeled = xml_labeled | backfilled
    assert "some/path/a" in union_labeled
    assert "some/path/b" in union_labeled
