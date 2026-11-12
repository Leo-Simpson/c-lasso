from types import SimpleNamespace

import pytest

from ..path_alg import up


def test_up_unknown_formulation_raises_value_error():
    param = SimpleNamespace(formulation="R5")
    with pytest.raises(ValueError, match="Unknown formulation R5"):
        up(param)
