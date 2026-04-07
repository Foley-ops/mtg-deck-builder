"""Rate limiter."""

import time


class RateLimiter:
    """Per-domain minimum interval between requests."""

    _last_call: dict[str, float] = {}

    @classmethod
    def wait(cls, domain, min_interval=0.1):
        now = time.time()
        elapsed = now - cls._last_call.get(domain, 0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        cls._last_call[domain] = time.time()
