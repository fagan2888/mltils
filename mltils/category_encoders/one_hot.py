
import category_encoders as ce
from .base import EncoderWrapper


class OneHotEncoder(EncoderWrapper):
    def __init__(self, cat_vars, *args, **kwargs):
        super().__init__(
            cat_vars=cat_vars,
            encoder_class=ce.one_hot.OneHotEncoder,
            suffix='ohe',
            *args, **kwargs
        )
