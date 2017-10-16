
import category_encoders as ce
from .base import EncoderWrapper


class SumEncoder(EncoderWrapper):
    def __init__(self, cat_vars, *args, **kwargs):
        super().__init__(
            cat_vars=cat_vars,
            encoder_class=ce.sum_coding.SumEncoder,
            suffix='sue',
            *args, **kwargs
        )
