
import category_encoders as ce
from .base import EncoderWrapper


class LeaveOneOutEncoder(EncoderWrapper):
    def __init__(self, cat_vars, *args, **kwargs):
        super().__init__(
            cat_vars=cat_vars,
            encoder_class=ce.leave_one_out.LeaveOneOutEncoder,
            suffix='loe',
            *args, **kwargs
        )
