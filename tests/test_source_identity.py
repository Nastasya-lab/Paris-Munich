import pytest

from weather_tmax_bot.temporal.source_identity import require_same_source
from weather_tmax_bot.utils.validation import SourceIdentityError


def test_source_mismatch_errors():
    with pytest.raises(SourceIdentityError):
        require_same_source("iem.metar", "awc.metar")
