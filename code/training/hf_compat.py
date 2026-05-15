from transformers.modeling_utils import PreTrainedModel


def patch_all_tied_weights_keys() -> None:
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    @property
    def all_tied_weights_keys(self):
        compat_value = getattr(self, "_compat_all_tied_weights_keys", None)
        if compat_value is not None:
            return compat_value
        tied = getattr(self, "_tied_weights_keys", None)
        if tied is None:
            return {}
        if isinstance(tied, dict):
            return tied
        if isinstance(tied, (list, tuple, set)):
            return {k: None for k in tied}
        return {}

    @all_tied_weights_keys.setter
    def all_tied_weights_keys(self, value):
        self._compat_all_tied_weights_keys = value

    PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys
