from __future__ import annotations


class WeatherTmaxError(RuntimeError):
    pass


class DataAvailabilityError(WeatherTmaxError):
    pass


class LeakageError(WeatherTmaxError):
    pass


class SourceIdentityError(WeatherTmaxError):
    pass
