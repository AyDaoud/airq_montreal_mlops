"""Resolve CKAN resource IDs to current download URLs."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.http import build_session
from src.data.sources import CKAN_BASE, Resource


class CkanError(RuntimeError):
    """Raised when a resource cannot be resolved."""


@dataclass(frozen=True)
class ResolvedResource:
    name: str
    url: str
    last_modified: str | None


def resolve(resource: Resource, session=None, timeout: int = 60) -> ResolvedResource:
    """Return the current download URL for ``resource``.

    Falls back to ``resource.fallback_url`` when the API cannot be reached.
    Raises :class:`CkanError` when there is no fallback to fall back to.
    """
    session = session or build_session()
    try:
        response = session.get(
            f"{CKAN_BASE}/package_show",
            params={"id": resource.dataset_id},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # network, DNS, JSON, HTTP status
        if resource.fallback_url:
            return ResolvedResource(resource.name, resource.fallback_url, None)
        raise CkanError(
            f"Could not resolve resource {resource.name} "
            f"({resource.resource_id}) and no fallback URL is configured"
        ) from exc

    for entry in payload.get("result", {}).get("resources", []):
        if entry.get("id") == resource.resource_id:
            url = entry.get("url")
            if not url:
                raise CkanError(f"Resource {resource.name} has no url field")
            return ResolvedResource(resource.name, url, entry.get("last_modified"))

    raise CkanError(
        f"Resource {resource.resource_id} not found in dataset {resource.dataset_id}"
    )
