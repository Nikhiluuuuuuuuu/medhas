import pytest
from medhas.retrieval.spreading_activation import SpreadingActivationEngine

def test_spreading_activation_struct():
    engine = SpreadingActivationEngine()
    assert engine.decay == 0.75
    assert engine.threshold == 0.15
