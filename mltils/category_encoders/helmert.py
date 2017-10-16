
import category_encoders as ce
from .base import EncoderWrapper


class HelmertEncoder(EncoderWrapper):
    def __init__(self, cat_vars, *args, **kwargs):
        super().__init__(
            cat_vars=cat_vars,
            encoder_class=ce.helmert.HelmertEncoder,
            suffix='hme',
            *args, **kwargs
        )
