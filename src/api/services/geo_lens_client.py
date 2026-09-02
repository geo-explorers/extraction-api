"""Thin client for geo-lens (github.com/geobrowser/geo-lens): cached Geo subgraphs with
vector / text / exact query strategies. This service only READS from it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

RETRYABLE = {429, 502, 503, 504}


class GeoLensError(Exception):
    pass


@dataclass(frozen=True)
class LensHit:
    id: str
    name: str
    score: float
    payload: dict


class GeoLensClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        timeout_s: float = 30.0,
        retries: int = 3,
    ):
        self.base_url = (base_url or settings.geo_lens_url).rstrip("/")
        self.api_key = api_key or settings.geo_lens_api_key
        if not self.api_key:
            raise GeoLensError("GEO_LENS_API_KEY is not configured")
        self.timeout_s = timeout_s
        self.retries = retries

    async def query(
        self,
        cache: str,
        strategy: str,
        input: dict[str, Any],
        *,
        k: int = 10,
        min_score: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[LensHit]:
        body: dict[str, Any] = {"strategy": strategy, "input": input, "k": k, "filters": filters or {}}
        if min_score is not None:
            body["min_score"] = min_score
        data = await self._post(f"/caches/{cache}/query", body)
        return [
            LensHit(id=h["id"], name=h.get("name") or "", score=float(h["score"]), payload=h.get("payload") or {})
            for h in data.get("hits", [])
        ]

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        delay = 1.0
        async with httpx.AsyncClient(timeout=self.timeout_s, headers={"X-API-Key": self.api_key}) as http:
            for attempt in range(1, self.retries + 1):
                try:
                    response = await http.post(self.base_url + path, json=body)
                except (httpx.TransportError, httpx.TimeoutException) as err:
                    if attempt == self.retries:
                        raise GeoLensError(f"geo-lens unreachable: {err!r}") from err
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if response.status_code in RETRYABLE and attempt < self.retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if response.status_code >= 400:
                    raise GeoLensError(f"geo-lens {path} -> HTTP {response.status_code}: {response.text[:300]}")
                return response.json()
        raise GeoLensError("geo-lens: no attempts made")
