"""External-source scrapers for the social-context pipeline.

Each scraper:
  * pulls a small batch of recent items from one platform,
  * normalizes them into `SourceSnapshot` objects,
  * never raises out — failures are logged and counted by the scheduler,
  * respects a per-source `TokenBucket` for rate limiting.

V1 sources shipped (PR #4): Reddit, NewsAPI.
V1 deferred (PR #6): Google Trends.
V1 disabled-by-default skeleton: X / Twitter.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, ClassVar

from .ratelimit import TokenBucket
from .schemas import IngestStats, SourceKind, SourceSnapshot

_log = logging.getLogger("cortyze.social_context.scraper")


class Scraper(ABC):
    """Abstract scraper. Implementations declare their `source` kind
    and the env var that toggles them on; the scheduler queries
    `enabled` before invoking `fetch`.
    """

    source: ClassVar[SourceKind]
    rate_limit_env: ClassVar[str] = ""
    rate_limit_default: ClassVar[float] = 30.0

    def __init__(self) -> None:
        rate = self._resolve_rate_limit()
        self._bucket = TokenBucket(rate_per_min=rate, burst=int(rate))

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether the scraper has the env / lib it needs to run.

        Implementations check API-key presence and library imports here.
        Returning False causes the scheduler to skip this scraper
        cleanly — no errors logged, no stats recorded.
        """

    @abstractmethod
    def fetch(self, *, limit: int = 25) -> list[SourceSnapshot]:
        """Pull a batch of recent items. Must catch its own errors and
        return what it can — never raise out of this method.
        """

    # ----------------------------------------------------------- helpers

    def _resolve_rate_limit(self) -> float:
        if not self.rate_limit_env:
            return self.rate_limit_default
        raw = os.environ.get(self.rate_limit_env)
        if not raw:
            return self.rate_limit_default
        try:
            return float(raw)
        except ValueError:
            _log.warning(
                "%s=%r not a float; using default %.1f/min",
                self.rate_limit_env,
                raw,
                self.rate_limit_default,
            )
            return self.rate_limit_default

    def _consume(self) -> bool:
        if self._bucket.try_consume(1):
            return True
        _log.info(
            "%s scraper rate-limited (bucket empty); skipping this pass",
            self.source,
        )
        return False


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------


class RedditScraper(Scraper):
    """Pulls recent top posts from a configurable list of subreddits.

    `REDDIT_SUBREDDITS` (comma-separated) defaults to a marketing-leaning
    list — the scheduler reloads this on each pass so operators can
    rotate the list without a redeploy.
    """

    source = "reddit"
    rate_limit_env = "REDDIT_RATE_LIMIT_PER_MIN"
    rate_limit_default = 30.0

    @property
    def enabled(self) -> bool:
        if not (
            os.environ.get("REDDIT_CLIENT_ID")
            and os.environ.get("REDDIT_CLIENT_SECRET")
        ):
            return False
        try:
            import praw  # noqa: F401  WPS433
        except ImportError:
            return False
        return True

    def fetch(self, *, limit: int = 25) -> list[SourceSnapshot]:
        if not self.enabled:
            return []
        if not self._consume():
            return []
        try:
            import praw
        except ImportError:
            return []
        try:
            reddit = praw.Reddit(
                client_id=os.environ["REDDIT_CLIENT_ID"],
                client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                user_agent=os.environ.get(
                    "REDDIT_USER_AGENT", "cortyze/0.1 social_context"
                ),
                check_for_async=False,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("praw client construct failed: %s", exc)
            return []

        subs_raw = os.environ.get(
            "REDDIT_SUBREDDITS",
            "marketing,advertising,videos,television,popular",
        )
        subs = [s.strip() for s in subs_raw.split(",") if s.strip()]
        out: list[SourceSnapshot] = []
        per_sub = max(1, limit // max(len(subs), 1))
        for sub in subs:
            try:
                for post in reddit.subreddit(sub).hot(limit=per_sub):
                    out.append(
                        SourceSnapshot(
                            source="reddit",
                            source_id=str(post.id),
                            title=getattr(post, "title", "") or "",
                            body=(getattr(post, "selftext", "") or "")[:2000],
                            url=f"https://www.reddit.com{getattr(post, 'permalink', '')}",
                            author=str(getattr(post, "author", "") or ""),
                            score=float(getattr(post, "score", 0) or 0),
                            extra={"subreddit": sub},
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "reddit subreddit=%s fetch failed: %s", sub, exc
                )
                continue
        return out


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------


class NewsScraper(Scraper):
    """Pulls top headlines from NewsAPI's `/v2/top-headlines` endpoint.

    Free tier: 100 req/day. Default polls 5 categories every 30 min →
    240 req/day, slightly over — operator can tune `NEWSAPI_CATEGORIES`
    or move to a paid tier. The 429 path lands in the rate limiter so
    we degrade gracefully when the cap is hit.
    """

    source = "news"
    rate_limit_env = "NEWSAPI_RATE_LIMIT_PER_MIN"
    rate_limit_default = 4.0

    _ENDPOINT = "https://newsapi.org/v2/top-headlines"

    @property
    def enabled(self) -> bool:
        if not os.environ.get("NEWSAPI_KEY"):
            return False
        try:
            import requests  # noqa: F401  WPS433
        except ImportError:
            return False
        return True

    def fetch(self, *, limit: int = 25) -> list[SourceSnapshot]:
        if not self.enabled:
            return []
        if not self._consume():
            return []
        try:
            import requests
        except ImportError:
            return []

        key = os.environ["NEWSAPI_KEY"]
        cats_raw = os.environ.get(
            "NEWSAPI_CATEGORIES",
            "business,entertainment,technology",
        )
        categories = [c.strip() for c in cats_raw.split(",") if c.strip()]
        out: list[SourceSnapshot] = []
        per_cat = max(1, limit // max(len(categories), 1))
        for cat in categories:
            try:
                resp = requests.get(
                    self._ENDPOINT,
                    params={
                        "category": cat,
                        "country": os.environ.get(
                            "NEWSAPI_COUNTRY", "us"
                        ),
                        "pageSize": per_cat,
                        "apiKey": key,
                    },
                    timeout=10.0,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("newsapi cat=%s GET failed: %s", cat, exc)
                continue
            if resp.status_code != 200:
                _log.warning(
                    "newsapi cat=%s returned HTTP %d", cat, resp.status_code
                )
                continue
            try:
                payload: dict[str, Any] = resp.json()
            except ValueError:
                _log.warning("newsapi cat=%s returned non-JSON", cat)
                continue
            for art in payload.get("articles", []) or []:
                url = art.get("url") or ""
                source_id = art.get("publishedAt", "") + "::" + url
                source_id_hash = hashlib.sha256(source_id.encode()).hexdigest()[:24]
                out.append(
                    SourceSnapshot(
                        source="news",
                        source_id=source_id_hash,
                        title=art.get("title") or "",
                        body=(art.get("description") or "")[:2000],
                        url=url or None,
                        author=art.get("author") or None,
                        score=0.0,
                        extra={
                            "category": cat,
                            "publishedAt": art.get("publishedAt"),
                            "source_name": (
                                (art.get("source") or {}).get("name") or ""
                            ),
                        },
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Google Trends (PR #6)
# ---------------------------------------------------------------------------


class GoogleTrendsScraper(Scraper):
    """Pulls "rising queries" + "trending searches" via `pytrends`.

    pytrends is unauthenticated but rate-limit-prone — Google returns
    429 and HTML CAPTCHA pages on heavy use. Default rate is conservative
    (3 calls/min); if you see a sustained 429 stream in logs, drop the
    rate further or pause the scraper via `TRENDS_SCRAPER_ENABLED=false`.
    """

    source = "trends"
    rate_limit_env = "PYTRENDS_RATE_LIMIT_PER_MIN"
    rate_limit_default = 3.0

    @property
    def enabled(self) -> bool:
        if (
            os.environ.get(
                "TRENDS_SCRAPER_ENABLED", "true"
            )
            .strip()
            .lower()
            == "false"
        ):
            return False
        try:
            from pytrends.request import TrendReq  # noqa: F401  WPS433
        except ImportError:
            return False
        return True

    def fetch(self, *, limit: int = 25) -> list[SourceSnapshot]:
        if not self.enabled:
            return []
        if not self._consume():
            return []
        try:
            from pytrends.request import TrendReq
        except ImportError:
            return []
        try:
            client = TrendReq(
                hl=os.environ.get("PYTRENDS_HL", "en-US"),
                tz=int(os.environ.get("PYTRENDS_TZ", "0")),
                requests_args={"timeout": 10},
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("pytrends client construct failed: %s", exc)
            return []

        out: list[SourceSnapshot] = []
        country = os.environ.get("PYTRENDS_COUNTRY", "united_states")
        try:
            df = client.trending_searches(pn=country)
        except Exception as exc:  # noqa: BLE001
            _log.warning("pytrends trending_searches failed: %s", exc)
            return []
        # `trending_searches` returns a single-column DataFrame of query
        # strings — there's no platform-stable id, so we hash the query
        # text + the day to dedupe.
        try:
            from datetime import date

            today = date.today().isoformat()
            for raw_query in df.iloc[:, 0].tolist()[:limit]:
                query = str(raw_query).strip()
                if not query:
                    continue
                source_id = hashlib.sha256(
                    f"{today}::{query.lower()}".encode()
                ).hexdigest()[:24]
                out.append(
                    SourceSnapshot(
                        source="trends",
                        source_id=source_id,
                        title=query,
                        body="",  # pytrends gives the query, not context
                        extra={"country": country, "as_of_date": today},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("pytrends result parse failed: %s", exc)
            return []
        return out


# ---------------------------------------------------------------------------
# X / Twitter — disabled-by-default skeleton (PR #4)
# ---------------------------------------------------------------------------


class XScraper(Scraper):
    """Tweepy-backed pull. Disabled by default — set `X_BEARER_TOKEN`
    AND `X_SCRAPER_ENABLED=true` to flip it on. We require the explicit
    enable flag because the X Basic tier is paid and rate-limited; we
    don't want a misconfigured deployment to start burning quota.
    """

    source = "x"
    rate_limit_env = "X_RATE_LIMIT_PER_MIN"
    rate_limit_default = 3.0

    @property
    def enabled(self) -> bool:
        if (
            os.environ.get("X_SCRAPER_ENABLED", "").strip().lower()
            != "true"
        ):
            return False
        if not os.environ.get("X_BEARER_TOKEN"):
            return False
        try:
            import tweepy  # noqa: F401  WPS433
        except ImportError:
            return False
        return True

    def fetch(self, *, limit: int = 25) -> list[SourceSnapshot]:  # pragma: no cover
        # Skeleton — implementation intentionally deferred. When the
        # team wants X data, fill this in following the Reddit pattern.
        del limit
        if not self.enabled:
            return []
        _log.info(
            "x scraper enabled but fetch() is a skeleton; returning []"
        )
        return []


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------


def all_scrapers() -> list[Scraper]:
    """Concrete scrapers, in execution order. The scheduler runs them
    concurrently via `asyncio.gather(return_exceptions=True)`.
    """
    return [
        RedditScraper(),
        NewsScraper(),
        GoogleTrendsScraper(),
        XScraper(),
    ]


def ingest_one(
    scraper: Scraper, *, limit: int = 25
) -> tuple[list[SourceSnapshot], IngestStats]:
    """Run a single scraper pass and return the results + stats blob.

    Stats are cumulative for the pass — node/edge counts are filled in
    by the graph builder downstream because the scraper doesn't see
    the graph.
    """
    started = time.monotonic()
    snaps: list[SourceSnapshot] = []
    error_count = 0
    if not scraper.enabled:
        return snaps, IngestStats(
            source=scraper.source,
            snapshots_ingested=0,
            errors=0,
            latency_ms=0,
        )
    try:
        snaps = scraper.fetch(limit=limit)
    except Exception as exc:  # noqa: BLE001
        _log.exception("scraper %s raised; counting as errored", scraper.source)
        error_count = 1
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return snaps, IngestStats(
        source=scraper.source,
        snapshots_ingested=len(snaps),
        errors=error_count,
        latency_ms=elapsed_ms,
        finished_at=datetime.now(timezone.utc),
    )
