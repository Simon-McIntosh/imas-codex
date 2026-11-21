"""Archive configuration checks using invented endpoint identities only."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from iter_codex.config.codac_uda import (
    archive_host_key,
    resolve_archive_endpoint,
    validate_codac_uda_config,
)
from iter_codex.config.models import CodacUDAConfig, FacilityConfig
from iter_codex.discovery.base.facility import _deep_merge


@pytest.fixture
def archive_config() -> dict:
    return {
        "setup_commands": ["export DEMO_CLIENT=enabled"],
        "python_command": "python3",
        "archives": [
            {"role": "fast", "host": "fast.example.invalid"},
            {"role": "slow", "host": "slow.example.invalid"},
            {"role": "operator", "host": "operators.example.invalid", "excluded": True},
        ],
        "readable_roles": ["fast", "slow"],
        "variable_patterns": ["DEMO:CURRENT-*"],
    }


def test_archive_configuration_round_trip(archive_config: dict) -> None:
    config = validate_codac_uda_config(archive_config)
    restored = validate_codac_uda_config(config.model_dump(mode="json"))
    assert restored == config
    assert restored.max_window_ns == 60_000_000_000
    assert restored.max_samples == 100_000
    assert restored.max_bytes == 8_388_608
    assert restored.setup_commands == ["export DEMO_CLIENT=enabled"]
    assert resolve_archive_endpoint(restored, "fast").host == "fast.example.invalid"
    assert resolve_archive_endpoint(restored, "slow").port == 3090


def test_operator_cannot_be_named_as_readable_role(archive_config: dict) -> None:
    archive_config["readable_roles"] = ["operator"]
    with pytest.raises(ValidationError, match="readable_roles"):
        FacilityConfig.model_validate(
            {
                "id": "demo",
                "name": "Demonstration",
                "machine": "DEMO",
                "data_systems": {"codac_uda": archive_config},
            }
        )


@pytest.mark.parametrize(
    "host", ["operators.example.invalid", "OPERATORS.other.invalid.", "operators"]
)
def test_readable_host_cannot_alias_reserved_endpoint(
    archive_config: dict, host: str
) -> None:
    archive_config["archives"][0]["host"] = host
    with pytest.raises(ValueError, match="reserved or excluded"):
        validate_codac_uda_config(archive_config)
    with pytest.raises(ValueError, match="reserved or excluded"):
        resolve_archive_endpoint(archive_config, "fast")


def test_operator_role_cannot_drop_exclusion(archive_config: dict) -> None:
    archive_config["archives"][2]["excluded"] = False
    with pytest.raises(ValueError, match="explicitly excluded"):
        validate_codac_uda_config(archive_config)


def test_duplicate_roles_are_ambiguous(archive_config: dict) -> None:
    archive_config["archives"].append(deepcopy(archive_config["archives"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_codac_uda_config(archive_config)


@pytest.mark.parametrize("role", ["operator", "unknown"])
def test_resolver_refuses_non_readable_role(archive_config: dict, role: str) -> None:
    with pytest.raises(ValueError, match="not readable"):
        resolve_archive_endpoint(archive_config, role)


@pytest.mark.parametrize("mutation", ["absent", "excluded"])
def test_readable_role_requires_available_endpoint(
    archive_config: dict, mutation: str
) -> None:
    if mutation == "absent":
        archive_config["archives"].pop(0)
    else:
        archive_config["archives"][0]["excluded"] = True
    with pytest.raises(ValueError, match="non-excluded endpoint"):
        validate_codac_uda_config(archive_config)


def test_public_config_validates_but_cannot_resolve_private_hosts() -> None:
    path = (
        Path(__file__).resolve().parents[2] / "iter_codex/config/facilities/iter.yaml"
    )
    public = yaml.safe_load(path.read_text())
    config = FacilityConfig.model_validate(public).data_systems.codac_uda
    assert config is not None
    validate_codac_uda_config(config)
    assert config.python_command == "python3"
    assert any("module load" in step for step in config.setup_commands)
    assert all(item.host is None for item in config.archives)
    with pytest.raises(ValueError, match="private operator endpoint exclusion"):
        resolve_archive_endpoint(config, "fast")


@pytest.mark.parametrize("missing_role", ["fast", "operator"])
def test_resolver_requires_private_endpoint_hosts(
    archive_config: dict, missing_role: str
) -> None:
    next(
        item for item in archive_config["archives"] if item["role"] == missing_role
    ).pop("host")
    with pytest.raises(ValueError, match="must be configured"):
        resolve_archive_endpoint(archive_config, "fast")


@pytest.mark.parametrize(
    "field", ["max_samples", "max_bytes", "max_window_ns", "timeout"]
)
def test_read_limits_must_be_positive(archive_config: dict, field: str) -> None:
    archive_config[field] = 0
    with pytest.raises(ValidationError):
        CodacUDAConfig.model_validate(archive_config)


@pytest.mark.parametrize("port", [0, 65536])
def test_endpoint_port_is_bounded(archive_config: dict, port: int) -> None:
    archive_config["archives"][0]["port"] = port
    with pytest.raises(ValidationError):
        CodacUDAConfig.model_validate(archive_config)


def test_ip_addresses_are_compared_exactly(archive_config: dict) -> None:
    archive_config["archives"][0]["host"] = "192.0.2.10"
    archive_config["archives"][2]["host"] = "192.0.2.11"
    assert resolve_archive_endpoint(archive_config, "fast").host == "192.0.2.10"


@pytest.mark.parametrize(
    ("read_host", "operator_host", "reserved"),
    [
        ("OPERATORS.example.invalid.", "operators.example.invalid", True),
        ("operators", "operators.example.invalid", True),
        ("archive.example.invalid", "operators.example.invalid", False),
        ("192.0.2.10", "192.0.2.10", True),
        ("192.0.2.10", "192.0.2.11", False),
        ("2001:db8::1", "2001:0db8:0:0:0:0:0:1", True),
        ("2001:db8::1", "2001:db8::2", False),
    ],
)
def test_config_owns_host_comparison_for_names_aliases_and_addresses(
    archive_config: dict, read_host: str, operator_host: str, reserved: bool
) -> None:
    assert (archive_host_key(read_host) == archive_host_key(operator_host)) is reserved
    archive_config["archives"][0]["host"] = read_host
    archive_config["archives"][2]["host"] = operator_host
    if reserved:
        with pytest.raises(ValueError, match="reserved or excluded"):
            resolve_archive_endpoint(archive_config, "fast")
    else:
        assert resolve_archive_endpoint(archive_config, "fast").host == read_host


@pytest.mark.parametrize("private_roles", [("fast",), ("fast", "slow", "operator")])
def test_private_archive_list_replacement_explains_role_redeclaration(
    archive_config: dict, private_roles: tuple[str, ...]
) -> None:
    public = {"data_systems": {"codac_uda": archive_config}}
    private = {
        "data_systems": {
            "codac_uda": {
                "archives": [
                    {"role": role, "host": f"{role}.example.invalid"}
                    for role in private_roles
                ]
            }
        }
    }
    merged = _deep_merge(public, private)["data_systems"]["codac_uda"]
    assert [item["role"] for item in merged["archives"]] == list(private_roles)
    assert len(archive_config["archives"]) == 3
    with pytest.raises(ValueError, match="private block must redeclare every role"):
        validate_codac_uda_config(merged)


def test_complete_private_archive_list_redeclares_roles_and_policy(
    archive_config: dict,
) -> None:
    public = deepcopy(archive_config)
    for endpoint in public["archives"]:
        endpoint.pop("host")
    merged = _deep_merge(public, {"archives": archive_config["archives"]})
    assert resolve_archive_endpoint(merged, "fast").host == "fast.example.invalid"
    assert resolve_archive_endpoint(merged, "slow").host == "slow.example.invalid"
    with pytest.raises(ValueError, match="not readable"):
        resolve_archive_endpoint(merged, "operator")
