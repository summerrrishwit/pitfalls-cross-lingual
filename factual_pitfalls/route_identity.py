"""Redacted route-identity binding for resumable model-backed runners."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

from dotenv import dotenv_values


ROUTE_IDENTITY_SCHEMA = "redacted-model-route-identity-v1"
SUPPORTED_MODEL_PROTOCOLS = frozenset({"anthropic", "openai", "openai_compatible"})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ROUTE_IDENTITY_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "provider_profile",
        "protocol",
        "requested_model",
        "expected_response_model",
        "normalized_base_url_sha256",
        "normalized_request_route_sha256",
        "credentials_or_endpoints_included",
        "route_identity_sha256",
    }
)
ROUTE_IDENTITY_SET_FIELDS = frozenset(
    {
        "schema_version",
        "record_count",
        "records",
        "credentials_or_endpoints_included",
        "route_identity_set_sha256",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_model_protocol(value: Any) -> str:
    """Return an exact supported protocol name or fail closed."""

    if not isinstance(value, str) or value not in SUPPORTED_MODEL_PROTOCOLS:
        raise ValueError("unsupported model route protocol")
    return value


def _api_root(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/messages", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def _normalized_endpoint_url(value: str) -> str:
    """Normalize an endpoint for hashing without returning it to an artifact."""

    parsed = urlparse(str(value).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("model route endpoint identity is invalid")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("model route endpoint must not embed credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("model route endpoint port is invalid") from error
    hostname = parsed.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    return urlunparse(
        (parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", "", "")
    )


def _resolved_base_url(
    config: Mapping[str, Any],
    env_path: Path,
    model_spec: Mapping[str, Any],
    *,
    resolved_env: Optional[Mapping[str, Any]] = None,
) -> Tuple[Mapping[str, Any], str]:
    profile_name = str(model_spec.get("provider_profile") or "").strip()
    profiles = config.get("provider_profiles")
    if not profile_name or not isinstance(profiles, dict):
        raise ValueError("model route provider profile is missing")
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown model route provider profile: {profile_name}")
    values = dict(resolved_env) if resolved_env is not None else dotenv_values(Path(env_path))
    base_url_env = profile.get("base_url_env")
    base_url = str(
        (values.get(base_url_env) if isinstance(base_url_env, str) else None)
        or profile.get("base_url")
        or ""
    ).strip()
    if not base_url:
        raise ValueError(f"model route endpoint is not configured: {profile_name}")
    return profile, base_url


def build_route_identity(
    *,
    config: Mapping[str, Any],
    env_path: Path,
    model_spec: Mapping[str, Any],
    expected_response_model: str,
    resolved_env: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a route binding that contains hashes, never endpoint or credential text."""

    model = str(model_spec.get("model") or "").strip()
    provider_profile = str(model_spec.get("provider_profile") or "").strip()
    expected = str(expected_response_model or "").strip()
    if not model or not provider_profile or not expected:
        raise ValueError("model route identity fields must be non-empty")
    profile, base_url = _resolved_base_url(
        config,
        Path(env_path),
        model_spec,
        resolved_env=resolved_env,
    )
    protocol = normalize_model_protocol(profile.get("protocol"))
    normalized_base = _normalized_endpoint_url(_api_root(base_url))
    route_suffix = "messages" if protocol == "anthropic" else "chat/completions"
    normalized_route = _normalized_endpoint_url(
        f"{normalized_base.rstrip('/')}/{route_suffix}"
    )
    record: Dict[str, Any] = {
        "schema_version": ROUTE_IDENTITY_SCHEMA,
        "provider_profile": provider_profile,
        "protocol": protocol,
        "requested_model": model,
        "expected_response_model": expected,
        "normalized_base_url_sha256": _sha256_text(normalized_base),
        "normalized_request_route_sha256": _sha256_text(normalized_route),
        "credentials_or_endpoints_included": False,
    }
    record["route_identity_sha256"] = _sha256_text(_canonical_json(record))
    return record


def build_route_identity_set(
    *,
    config: Mapping[str, Any],
    env_path: Path,
    routes: Sequence[Tuple[Mapping[str, Any], str]],
    resolved_env: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    records = [
        build_route_identity(
            config=config,
            env_path=env_path,
            model_spec=model_spec,
            expected_response_model=expected_response_model,
            resolved_env=resolved_env,
        )
        for model_spec, expected_response_model in routes
    ]
    keys = [
        (str(record["provider_profile"]), str(record["requested_model"]))
        for record in records
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model route identity key")
    identity_set = {
        "schema_version": "redacted-model-route-identity-set-v1",
        "record_count": len(records),
        "records": records,
        "credentials_or_endpoints_included": False,
    }
    identity_set["route_identity_set_sha256"] = _sha256_text(
        _canonical_json(identity_set)
    )
    return identity_set


def validate_route_identity_set(
    identity_set: Mapping[str, Any],
    *,
    expected_routes: Sequence[Tuple[str, str, str]],
    expected_protocols: Optional[Mapping[str, str]] = None,
) -> Sequence[Dict[str, Any]]:
    """Validate a persisted redacted route set and its exact model identities."""

    if set(identity_set) != ROUTE_IDENTITY_SET_FIELDS:
        raise ValueError("model route identity set fields are invalid")
    if identity_set.get("schema_version") != "redacted-model-route-identity-set-v1":
        raise ValueError("model route identity set schema is unsupported")
    records = identity_set.get("records")
    if not isinstance(records, list) or identity_set.get("record_count") != len(records):
        raise ValueError("model route identity record count is invalid")
    if identity_set.get("credentials_or_endpoints_included") is not False:
        raise ValueError("model route identity set is not redacted")
    payload = {
        key: value
        for key, value in identity_set.items()
        if key != "route_identity_set_sha256"
    }
    if identity_set.get("route_identity_set_sha256") != _sha256_text(
        _canonical_json(payload)
    ):
        raise ValueError("model route identity set SHA is stale")

    validated: list[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != ROUTE_IDENTITY_RECORD_FIELDS:
            raise ValueError("model route identity record fields are invalid")
        if record.get("schema_version") != ROUTE_IDENTITY_SCHEMA:
            raise ValueError("model route identity record schema is unsupported")
        if record.get("credentials_or_endpoints_included") is not False:
            raise ValueError("model route identity record is not redacted")
        for field in ("normalized_base_url_sha256", "normalized_request_route_sha256"):
            if not isinstance(record.get(field), str) or not SHA256_RE.fullmatch(
                str(record[field])
            ):
                raise ValueError(f"model route identity {field} is invalid")
        record_payload = {
            key: value for key, value in record.items() if key != "route_identity_sha256"
        }
        if record.get("route_identity_sha256") != _sha256_text(
            _canonical_json(record_payload)
        ):
            raise ValueError("model route identity record SHA is stale")
        protocol = normalize_model_protocol(record.get("protocol"))
        provider_profile = str(record.get("provider_profile") or "")
        if expected_protocols is not None and protocol != expected_protocols.get(
            provider_profile
        ):
            raise ValueError("model route identity protocol differs from config")
        validated.append(dict(record))

    observed_routes = [
        (
            str(record.get("provider_profile") or ""),
            str(record.get("requested_model") or ""),
            str(record.get("expected_response_model") or ""),
        )
        for record in validated
    ]
    normalized_expected = [tuple(map(str, route)) for route in expected_routes]
    if observed_routes != normalized_expected:
        raise ValueError("model route identities differ from the frozen model contract")
    keys = [(provider, requested) for provider, requested, _expected in observed_routes]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model route identity key")
    if "://" in _canonical_json(identity_set):
        raise ValueError("model route identity contains a plaintext endpoint")
    return validated


class RouteIdentityGuard:
    """Fail closed if a request tries to leave the route set frozen at startup."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        env_path: Path,
        identity_set: Mapping[str, Any],
        resolved_env: Optional[Mapping[str, Any]] = None,
        router: Optional[Any] = None,
    ) -> None:
        records = identity_set.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("model route identity set is empty")
        self.config = config
        self.env_path = Path(env_path)
        self.resolved_env = resolved_env
        self.router = router
        self.expected: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("model route identity record is invalid")
            key = (
                str(record["provider_profile"]),
                str(record["requested_model"]),
            )
            if key in self.expected:
                raise ValueError("duplicate model route identity key")
            self.expected[key] = dict(record)

    def assert_current_route(self, model_spec: Mapping[str, Any]) -> None:
        key = (
            str(model_spec.get("provider_profile") or ""),
            str(model_spec.get("model") or ""),
        )
        expected = self.expected.get(key)
        if expected is None:
            from factual_pitfalls.perturbation import StaticProxyIdentityError

            raise StaticProxyIdentityError(f"unfrozen_model_route:{key[0]}:{key[1]}")
        current_config = getattr(self.router, "config", self.config)
        if not isinstance(current_config, Mapping):
            current_config = self.config
        current_env = getattr(self.router, "values", self.resolved_env)
        if not isinstance(current_env, Mapping):
            current_env = self.resolved_env
        current = build_route_identity(
            config=current_config,
            env_path=self.env_path,
            model_spec=model_spec,
            expected_response_model=str(expected["expected_response_model"]),
            resolved_env=current_env,
        )
        if current != expected:
            from factual_pitfalls.perturbation import StaticProxyIdentityError

            raise StaticProxyIdentityError(f"model_route_identity_drift:{key[0]}:{key[1]}")


def attach_route_identity_guard(
    router: Any,
    *,
    config: Mapping[str, Any],
    env_path: Path,
    identity_set: Mapping[str, Any],
) -> None:
    """Attach the guard used by ModelRouter.request_json before every attempt."""

    resolved_env = getattr(router, "values", None)
    router.static_proxy_runtime_identity_guard = RouteIdentityGuard(
        config=config,
        env_path=env_path,
        identity_set=identity_set,
        resolved_env=resolved_env if isinstance(resolved_env, Mapping) else None,
        router=router,
    )
