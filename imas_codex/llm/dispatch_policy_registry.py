"""Strict atomic registry ownership for typed LLM dispatch policies."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from imas_codex.llm.callsite_registry import get_callsite_registration

DISPATCH_POLICY_DIR = Path(__file__).parent / "config" / "dispatch_policies"
_CHANNEL_NAMES = frozenset(
    {
        "source_facts",
        "approved_resolutions",
        "reviewer_intent",
        "comparators",
        "provenance",
        "batch_comparators",
    }
)


class DispatchPolicyRegistryError(ValueError):
    """A trusted dispatch-policy resource is absent, invalid, or ambiguous."""


@dataclass(frozen=True, slots=True)
class TemplateRoleSpec:
    """One template in the exact message-role order owned by a policy."""

    role: str
    name: str
    source_version: str


@dataclass(frozen=True, slots=True)
class ClaimChannelSpec:
    """Exact kinds and scopes admitted to one typed context channel."""

    channel: str
    kinds: frozenset[str]
    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class StaticProviderSpec:
    """Versioned schema, grammar, or other immutable provider snapshot."""

    name: str
    kind: str
    source_version: str


@dataclass(frozen=True, slots=True)
class AttachmentPolicySpec:
    """Registry limits for multimodal content carried by one request."""

    allowed_media_types: frozenset[str] = frozenset()
    max_count: int = 0
    max_bytes_each: int = 0
    max_bytes_total: int = 0
    max_width: int = 0
    max_height: int = 0


@dataclass(frozen=True, slots=True)
class DispatchPolicySpec:
    """Complete immutable authority, request, and spend policy for a callsite."""

    policy_id: str
    source_version: str
    callsite_id: str
    service: str
    seat: str
    task_kind: str
    templates: tuple[TemplateRoleSpec, ...]
    response_model_path: str
    model_section: str
    tokenizer_path: str
    tokenizer_key: str
    identifier_pattern: str
    channels: tuple[ClaimChannelSpec, ...]
    required_obligations: frozenset[str]
    static_providers: tuple[StaticProviderSpec, ...]
    max_input_tokens: int
    max_output_tokens: int
    max_attempts: int
    max_context_bytes: int
    maximum_cost_exposure: float
    attachment_policy: AttachmentPolicySpec = AttachmentPolicySpec()
    candidate_source_path: str | None = None
    temperature: float | None = None
    timeout: int | None = None
    reasoning_effort: str | None = None
    zero_cost_local: bool = False
    require_usage: bool = True
    supported: bool = True
    closure_blocker: str | None = None

    def channel(self, name: str) -> ClaimChannelSpec:
        """Return one exact channel policy or fail closed."""
        matches = [channel for channel in self.channels if channel.channel == name]
        if len(matches) != 1:
            raise DispatchPolicyRegistryError(
                f"Policy {self.policy_id!r} requires one {name!r} channel; "
                f"found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class ResolvedDispatchPolicy:
    """Trusted policy plus the exact runtime objects named by its resource."""

    spec: DispatchPolicySpec
    model: str
    response_model: type[BaseModel]
    token_counter: Callable[[Any], int]


@dataclass(frozen=True, slots=True)
class DispatchRegistrySnapshot:
    """Policies and blockers validated together before either is published."""

    policies: Mapping[str, DispatchPolicySpec]
    blockers: Mapping[str, str]


class _StrictResource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _TemplateResource(_StrictResource):
    role: Literal["system", "user"]
    name: str
    source_version: str


class _ChannelResource(_StrictResource):
    kinds: list[str]
    scopes: list[str]


class _ProviderResource(_StrictResource):
    name: str
    kind: str
    source_version: str


class _AttachmentResource(_StrictResource):
    allowed_media_types: list[str] = Field(default_factory=list)
    max_count: int = Field(default=0, ge=0)
    max_bytes_each: int = Field(default=0, ge=0)
    max_bytes_total: int = Field(default=0, ge=0)
    max_width: int = Field(default=0, ge=0)
    max_height: int = Field(default=0, ge=0)


class _PolicyResource(_StrictResource):
    policy_id: str
    source_version: str
    callsite_id: str
    service: str
    seat: str
    task_kind: str
    templates: list[_TemplateResource]
    response_model: str
    model_section: str
    tokenizer: str
    tokenizer_key: str
    identifier_pattern: str
    channels: dict[str, _ChannelResource]
    required_obligations: list[str]
    static_providers: list[_ProviderResource]
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_attempts: int = Field(gt=0)
    max_context_bytes: int = Field(gt=0)
    maximum_cost_exposure: float = Field(ge=0)
    attachments: _AttachmentResource = Field(default_factory=_AttachmentResource)
    candidate_source: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    timeout: int | None = Field(default=None, gt=0)
    reasoning_effort: (
        Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    zero_cost_local: bool = False
    require_usage: bool = True
    supported: bool = True
    closure_blocker: str | None = None


class _PolicyFile(_StrictResource):
    policies: list[_PolicyResource]


class _BlockerResource(_StrictResource):
    callsite_id: str
    closure_blocker: str


class _BlockerFile(_StrictResource):
    unsupported: list[_BlockerResource]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DispatchPolicyRegistryError(
                f"Dispatch resource contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _strict_json_load(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def _require_string(value: str, location: str) -> str:
    if not value.strip():
        raise DispatchPolicyRegistryError(f"{location} must be a non-blank string")
    return value


def _string_set(value: Sequence[str], location: str) -> frozenset[str]:
    if any(not item.strip() for item in value):
        raise DispatchPolicyRegistryError(f"{location} must contain non-blank strings")
    if len(value) != len(set(value)):
        raise DispatchPolicyRegistryError(f"{location} contains duplicates")
    return frozenset(value)


def _import_object(path: str) -> Any:
    module_name, separator, qualname = path.partition(":")
    if not separator or not module_name or not qualname:
        raise DispatchPolicyRegistryError(
            f"Registered object path must use module:qualname syntax: {path!r}"
        )
    try:
        value: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise DispatchPolicyRegistryError(
            f"Cannot resolve registered object {path!r}"
        ) from exc
    return value


def _candidate_models(path: str) -> tuple[str, ...]:
    source = _import_object(path)
    if not callable(source):
        raise DispatchPolicyRegistryError(f"Candidate source {path!r} is not callable")
    values = source()
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise DispatchPolicyRegistryError(
            f"Candidate source {path!r} must return a finite sequence"
        )
    candidates = tuple(values)
    if not candidates or any(
        not isinstance(item, str) or not item.strip() for item in candidates
    ):
        raise DispatchPolicyRegistryError(
            f"Candidate source {path!r} must return non-blank model strings"
        )
    if len(candidates) != len(set(candidates)):
        raise DispatchPolicyRegistryError(
            f"Candidate source {path!r} returned duplicate models"
        )
    return candidates


def _load_spec(data: _PolicyResource, location: str) -> DispatchPolicySpec:
    if not data.templates:
        raise DispatchPolicyRegistryError(f"{location}.templates must not be empty")
    roles = [template.role for template in data.templates]
    if roles[0] != "system":
        raise DispatchPolicyRegistryError(
            f"{location}.templates must start with system"
        )
    first_user = next(
        (index for index, role in enumerate(roles) if role == "user"), None
    )
    if first_user is not None and "system" in roles[first_user:]:
        raise DispatchPolicyRegistryError(
            f"{location}.templates cannot place system roles after user roles"
        )
    template_ids = [
        (item.role, item.name, item.source_version) for item in data.templates
    ]
    if len(template_ids) != len(set(template_ids)):
        raise DispatchPolicyRegistryError(f"{location}.templates contains duplicates")
    if set(data.channels) != _CHANNEL_NAMES:
        raise DispatchPolicyRegistryError(
            f"{location}.channels must be exactly {sorted(_CHANNEL_NAMES)}"
        )
    provider_ids = [
        (item.name, item.kind, item.source_version) for item in data.static_providers
    ]
    if len(provider_ids) != len(set(provider_ids)):
        raise DispatchPolicyRegistryError(
            f"{location}.static_providers contains duplicates"
        )
    attachment = data.attachments
    if attachment.max_count == 0:
        if any(
            (
                attachment.allowed_media_types,
                attachment.max_bytes_each,
                attachment.max_bytes_total,
                attachment.max_width,
                attachment.max_height,
            )
        ):
            raise DispatchPolicyRegistryError(
                f"{location}.attachments must be empty when max_count is zero"
            )
    elif (
        not attachment.allowed_media_types
        or attachment.max_bytes_each <= 0
        or attachment.max_bytes_total <= 0
        or attachment.max_width <= 0
        or attachment.max_height <= 0
    ):
        raise DispatchPolicyRegistryError(
            f"{location}.attachments requires media and positive bounds"
        )
    try:
        re.compile(data.identifier_pattern)
    except re.error as exc:
        raise DispatchPolicyRegistryError(
            f"{location}.identifier_pattern is invalid"
        ) from exc
    if data.model_section != data.seat:
        raise DispatchPolicyRegistryError(
            f"{location}.model_section must exactly match seat"
        )
    spec = DispatchPolicySpec(
        policy_id=_require_string(data.policy_id, f"{location}.policy_id"),
        source_version=_require_string(
            data.source_version, f"{location}.source_version"
        ),
        callsite_id=_require_string(data.callsite_id, f"{location}.callsite_id"),
        service=_require_string(data.service, f"{location}.service"),
        seat=_require_string(data.seat, f"{location}.seat"),
        task_kind=_require_string(data.task_kind, f"{location}.task_kind"),
        templates=tuple(
            TemplateRoleSpec(
                item.role,
                _require_string(item.name, f"{location}.templates.name"),
                _require_string(
                    item.source_version, f"{location}.templates.source_version"
                ),
            )
            for item in data.templates
        ),
        response_model_path=_require_string(
            data.response_model, f"{location}.response_model"
        ),
        model_section=data.model_section,
        tokenizer_path=_require_string(data.tokenizer, f"{location}.tokenizer"),
        tokenizer_key=_require_string(data.tokenizer_key, f"{location}.tokenizer_key"),
        identifier_pattern=data.identifier_pattern,
        channels=tuple(
            ClaimChannelSpec(
                channel=name,
                kinds=_string_set(value.kinds, f"{location}.channels.{name}.kinds"),
                scopes=_string_set(value.scopes, f"{location}.channels.{name}.scopes"),
            )
            for name, value in data.channels.items()
        ),
        required_obligations=_string_set(
            data.required_obligations, f"{location}.required_obligations"
        ),
        static_providers=tuple(
            StaticProviderSpec(item.name, item.kind, item.source_version)
            for item in data.static_providers
        ),
        max_input_tokens=data.max_input_tokens,
        max_output_tokens=data.max_output_tokens,
        max_attempts=data.max_attempts,
        max_context_bytes=data.max_context_bytes,
        maximum_cost_exposure=data.maximum_cost_exposure,
        attachment_policy=AttachmentPolicySpec(
            allowed_media_types=_string_set(
                attachment.allowed_media_types,
                f"{location}.attachments.allowed_media_types",
            ),
            max_count=attachment.max_count,
            max_bytes_each=attachment.max_bytes_each,
            max_bytes_total=attachment.max_bytes_total,
            max_width=attachment.max_width,
            max_height=attachment.max_height,
        ),
        candidate_source_path=data.candidate_source,
        temperature=data.temperature,
        timeout=data.timeout,
        reasoning_effort=data.reasoning_effort,
        zero_cost_local=data.zero_cost_local,
        require_usage=data.require_usage,
        supported=data.supported,
        closure_blocker=data.closure_blocker,
    )
    registration = get_callsite_registration(spec.callsite_id)
    response_model = _import_object(spec.response_model_path)
    if not isinstance(response_model, type) or not issubclass(
        response_model, BaseModel
    ):
        raise DispatchPolicyRegistryError(
            f"Response model {spec.response_model_path!r} is not a Pydantic model"
        )
    if registration.response_model_identity != spec.response_model_path:
        raise DispatchPolicyRegistryError(
            f"{location}.response_model differs from the registered response identity"
        )
    tokenizer = _import_object(spec.tokenizer_path)
    if not callable(tokenizer):
        raise DispatchPolicyRegistryError(
            f"Tokenizer {spec.tokenizer_path!r} is not callable"
        )
    if spec.candidate_source_path is not None:
        _candidate_models(spec.candidate_source_path)
    return spec


def load_dispatch_registry(
    directory: Path = DISPATCH_POLICY_DIR,
) -> DispatchRegistrySnapshot:
    """Load strict policies and blockers atomically into one snapshot."""
    policies: dict[str, DispatchPolicySpec] = {}
    blockers: dict[str, str] = {}
    policy_ids: set[str] = set()
    if not directory.exists():
        return DispatchRegistrySnapshot(
            MappingProxyType(policies), MappingProxyType(blockers)
        )
    for path in sorted(directory.glob("*.json")):
        try:
            payload_data = _strict_json_load(path.read_text())
            if path.name.endswith(".blocked.json"):
                payload = _BlockerFile.model_validate(payload_data)
                for index, entry in enumerate(payload.unsupported):
                    location = f"{path}:unsupported[{index}]"
                    callsite_id = _require_string(
                        entry.callsite_id, f"{location}.callsite_id"
                    )
                    reason = _require_string(
                        entry.closure_blocker, f"{location}.closure_blocker"
                    )
                    if callsite_id in blockers:
                        raise DispatchPolicyRegistryError(
                            f"Duplicate typed closure blocker for {callsite_id!r}"
                        )
                    get_callsite_registration(callsite_id)
                    blockers[callsite_id] = reason
            else:
                payload = _PolicyFile.model_validate(payload_data)
                for index, entry in enumerate(payload.policies):
                    spec = _load_spec(entry, f"{path}:policies[{index}]")
                    if spec.callsite_id in policies:
                        raise DispatchPolicyRegistryError(
                            f"Duplicate typed dispatch policy for {spec.callsite_id!r}"
                        )
                    if spec.policy_id in policy_ids:
                        raise DispatchPolicyRegistryError(
                            f"Duplicate typed policy id {spec.policy_id!r}"
                        )
                    policy_ids.add(spec.policy_id)
                    policies[spec.callsite_id] = spec
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise DispatchPolicyRegistryError(
                f"Cannot load strict dispatch resource {path}: {exc}"
            ) from exc
    collisions = set(policies) & set(blockers)
    if collisions:
        raise DispatchPolicyRegistryError(
            f"Typed callsites cannot be both active and blocked: {sorted(collisions)}"
        )
    return DispatchRegistrySnapshot(
        MappingProxyType(policies), MappingProxyType(blockers)
    )


def load_dispatch_policy_registry(
    directory: Path = DISPATCH_POLICY_DIR,
) -> Mapping[str, DispatchPolicySpec]:
    """Return policies from one atomically validated registry snapshot."""
    return load_dispatch_registry(directory).policies


def load_dispatch_closure_blockers(
    directory: Path = DISPATCH_POLICY_DIR,
) -> Mapping[str, str]:
    """Return blockers from one atomically validated registry snapshot."""
    return load_dispatch_registry(directory).blockers


def _resolve_spec(
    spec: DispatchPolicySpec,
    candidate_model: str | None,
) -> ResolvedDispatchPolicy:
    from imas_codex.settings import get_model

    if not spec.supported:
        reason = spec.closure_blocker or "route is explicitly unsupported"
        raise DispatchPolicyRegistryError(f"Callsite {spec.callsite_id!r}: {reason}")
    if spec.model_section != spec.seat:
        raise DispatchPolicyRegistryError(
            f"Policy {spec.policy_id!r} model section differs from its seat"
        )
    model = get_model(spec.model_section)
    if candidate_model is not None:
        if spec.candidate_source_path is None:
            raise DispatchPolicyRegistryError(
                f"Policy {spec.policy_id!r} does not permit a candidate model"
            )
        if candidate_model not in _candidate_models(spec.candidate_source_path):
            raise DispatchPolicyRegistryError(
                f"Candidate model is outside {spec.candidate_source_path!r}"
            )
        model = candidate_model
    response_model = _import_object(spec.response_model_path)
    token_counter = _import_object(spec.tokenizer_path)
    return ResolvedDispatchPolicy(spec, model, response_model, token_counter)


def resolve_dispatch_policy(
    callsite_id: str,
    *,
    candidate_model: str | None = None,
    registry: Mapping[str, DispatchPolicySpec] | None = None,
) -> ResolvedDispatchPolicy:
    """Resolve one trusted policy and only its explicitly permitted model axis."""
    active = DISPATCH_POLICY_REGISTRY if registry is None else registry
    spec = active.get(callsite_id)
    if spec is None:
        blocker = DISPATCH_CLOSURE_BLOCKERS.get(callsite_id)
        if blocker:
            raise DispatchPolicyRegistryError(
                f"Callsite {callsite_id!r} is typed-unsupported: {blocker}"
            )
        raise DispatchPolicyRegistryError(
            f"Callsite {callsite_id!r} has no typed-ready dispatch policy"
        )
    return _resolve_spec(spec, candidate_model)


def policy_registry_closure(
    observed_calls: Iterable[Any],
    *,
    registry: Mapping[str, DispatchPolicySpec] | None = None,
) -> tuple[int, int]:
    """Return legacy/typed counts and require every typed policy to resolve."""
    active = DISPATCH_POLICY_REGISTRY if registry is None else registry
    legacy = 0
    typed = 0
    for call in observed_calls:
        if getattr(call, "transition_kind", "legacy") == "typed":
            typed += 1
            callsite_id = getattr(call, "callsite_id", None)
            spec = active.get(callsite_id)
            if spec is None:
                raise DispatchPolicyRegistryError(
                    f"Typed expression has no policy: {callsite_id!r}"
                )
            _resolve_spec(spec, None)
        else:
            legacy += 1
    return legacy, typed


_DISPATCH_REGISTRY_SNAPSHOT = load_dispatch_registry()
DISPATCH_POLICY_REGISTRY = _DISPATCH_REGISTRY_SNAPSHOT.policies
DISPATCH_CLOSURE_BLOCKERS = _DISPATCH_REGISTRY_SNAPSHOT.blockers


__all__ = [
    "AttachmentPolicySpec",
    "ClaimChannelSpec",
    "DISPATCH_POLICY_DIR",
    "DISPATCH_POLICY_REGISTRY",
    "DISPATCH_CLOSURE_BLOCKERS",
    "DispatchPolicyRegistryError",
    "DispatchPolicySpec",
    "DispatchRegistrySnapshot",
    "ResolvedDispatchPolicy",
    "StaticProviderSpec",
    "TemplateRoleSpec",
    "load_dispatch_registry",
    "load_dispatch_policy_registry",
    "load_dispatch_closure_blockers",
    "policy_registry_closure",
    "resolve_dispatch_policy",
]
