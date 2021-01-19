from . import encoder
from easyvvuq.encoders import BaseEncoder

def test_create():
    bout_encoder = encoder.BOUTEncoder()

    assert isinstance(bout_encoder, BaseEncoder)
