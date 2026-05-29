from __future__ import annotations

from weather_tmax_bot.utils.validation import DataAvailabilityError


def fetch_mosmix_issued_archive(*_args, **_kwargs):
    raise DataAvailabilityError(
        "MOSMIX is supported only when an honest issued forecast archive is supplied or accumulated operationally."
    )
