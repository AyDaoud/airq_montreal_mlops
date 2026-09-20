"""Registry of Montreal open-data resources used by this project.

Resource IDs are stable; download URLs are not (the stations file embeds a
revision date in its filename). Always resolve through :mod:`src.data.ckan`.
"""

from __future__ import annotations

from dataclasses import dataclass

CKAN_BASE = "https://donnees.montreal.ca/api/3/action"


@dataclass(frozen=True)
class Resource:
    """One CKAN resource.

    ``fallback_url`` is used only if the CKAN API is unreachable. It is
    ``None`` where the filename is known to change between revisions.
    """

    name: str
    dataset_id: str
    resource_id: str
    fallback_url: str | None = None


_IQA_DATASET = "547b8052-1710-4d69-8760-beaa3aa35ec6"
_RT_DATASET = "3e9f7b96-3f25-4404-a5ad-22d9a31060e6"
_STATIONS_DATASET = "ae01f7f3-4d69-404a-9be1-74abfdc96571"

IQA_HISTORICAL: tuple[Resource, ...] = (
    Resource(
        name="iqa_2022_2024",
        dataset_id=_IQA_DATASET,
        resource_id="0c325562-e742-4e8e-8c36-971f3c9e58cd",
        fallback_url=(
            f"https://donnees.montreal.ca/dataset/{_IQA_DATASET}/resource/"
            "0c325562-e742-4e8e-8c36-971f3c9e58cd/download/"
            "rsqa-indice-qualite-air-2022-2024.csv"
        ),
    ),
    Resource(
        name="iqa_2025_2027",
        dataset_id=_IQA_DATASET,
        resource_id="6cf08815-49d2-4d2f-a400-ce36ee52b0fc",
        fallback_url=(
            f"https://donnees.montreal.ca/dataset/{_IQA_DATASET}/resource/"
            "6cf08815-49d2-4d2f-a400-ce36ee52b0fc/download/"
            "rsqa-indice-qualite-air-2025-2027.csv"
        ),
    ),
)

IQA_REALTIME = Resource(
    name="iqa_realtime",
    dataset_id=_RT_DATASET,
    resource_id="6554355e-63d1-4a01-a268-91e0763c3606",
    fallback_url=(
        f"https://donnees.montreal.ca/dataset/{_RT_DATASET}/resource/"
        "6554355e-63d1-4a01-a268-91e0763c3606/download/iqa-by-station.csv"
    ),
)

# No fallback: the filename carries a revision date and changes (decision D8).
STATIONS = Resource(
    name="stations",
    dataset_id=_STATIONS_DATASET,
    resource_id="29db5545-89a4-4e4a-9e95-05aa6dc2fd80",
    fallback_url=None,
)

# Station IDs that actually report IQA, verified 2026-09-14.
KNOWN_STATION_IDS: tuple[int, ...] = (3, 6, 17, 28, 31, 50, 55, 66, 80, 99, 103)

POLLUTANTS: tuple[str, ...] = ("PM", "O3", "NO2", "SO2", "CO")

# Montreal IQA scale: 1-25 Bon, 26-50 Acceptable, >50 Mauvais.
IQA_POOR_THRESHOLD = 50
