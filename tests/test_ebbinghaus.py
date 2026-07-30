import pytest
from medhas.consolidation.ebbinghaus import EbbinghausScrubber

def test_scrubber_init():
    scrubber = EbbinghausScrubber()
    assert scrubber.threshold == 0.10
