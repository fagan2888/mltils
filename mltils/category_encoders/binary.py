
import category_encoders as ce
from .base import EncoderWrapper


class BinaryEncoder(EncoderWrapper):
    def __init__(self, cat_vars, *args, **kwargs):
        super().__init__(
            cat_vars=cat_vars,
            encoder_class=ce.binary.BinaryEncoder,
            suffix='bie',
            *args, **kwargs
        )
