import pytest
import torch
from assertpy import assert_that
from torch_md.mlip.transformers import SimpleMLIP


class TestSimpleMLIP:
    @pytest.mark.parametrize(
        "input_tensor, expected_shape",
        [
            (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), torch.Size([])),
            (torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]), torch.Size([1])),
            (
                torch.tensor(
                    [
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    ]
                ),
                torch.Size([2]),
            ),
        ],
    )
    def test_forward(self, input_tensor: torch.Tensor, expected_shape: torch.Size):
        model = SimpleMLIP(d_model=8)
        output = model.forward(input_tensor)
        assert_that(output.shape).is_equal_to(expected_shape)
