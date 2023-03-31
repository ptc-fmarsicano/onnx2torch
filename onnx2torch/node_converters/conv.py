__all__ = []

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.padding import onnx_auto_pad_to_torch_padding

import math


class SameUpperConv(nn.Conv2d):
    def __init__(self, **kwargs):
        kwargs.pop("padding")
        super().__init__(**kwargs)

    def forward(self, x):
        return self.same_padding(x, self.weight.shape[2:4], self.stride)

    def same_padding(self, x, kernel_size, strides):
        input_h, input_w = x.shape[2:4]
        (filter_h, filter_w) = kernel_size

        output_h = int(math.ceil(float(input_h) / float(strides[0])))
        output_w = int(math.ceil(float(input_w) / float(strides[1])))

        pad_along_height = max((output_h - 1) * strides[0] + filter_h - input_h, 0)
        pad_along_width = max((output_w - 1) * strides[1] + filter_w - input_w, 0)
        pad_bottom = pad_along_height // 2
        pad_top = pad_along_height - pad_bottom
        pad_right = pad_along_width // 2
        pad_left = pad_along_width - pad_right
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        return super().forward(x)


_CONV_CLASS_FROM_SPATIAL_RANK = {
    ("Conv", 1): nn.Conv1d,
    ("Conv", 2): nn.Conv2d,
    ("Conv", 3): nn.Conv3d,
    ("ConvTranspose", 1): nn.ConvTranspose1d,
    ("ConvTranspose", 2): nn.ConvTranspose2d,
    ("ConvTranspose", 3): nn.ConvTranspose3d,
}


@add_converter(operation_type="Conv", version=1)
@add_converter(operation_type="Conv", version=11)
@add_converter(operation_type="ConvTranspose", version=1)
@add_converter(operation_type="ConvTranspose", version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    weights_value_name = node.input_values[1]
    weights = graph.initializers[weights_value_name]
    weights = weights.to_torch()
    if len(node.input_values) == 3:
        bias_value_name = node.input_values[2]
        bias = graph.initializers[bias_value_name]
        bias = bias.to_torch()
    else:
        bias = None

    op_type = node.operation_type
    spatial_rank = len(weights.shape) - 2
    try:
        conv_class = _CONV_CLASS_FROM_SPATIAL_RANK[op_type, spatial_rank]
    except KeyError as exc:
        raise NotImplementedError(
            f"Convolution operation with spatial rank == {spatial_rank} is not implemented"
        ) from exc

    node_attributes = node.attributes
    padding, input_padding_module = onnx_auto_pad_to_torch_padding(
        onnx_padding=node_attributes.get("pads", [0] * spatial_rank * 2),
        auto_pad=node_attributes.get("auto_pad", "NOTSET"),
    )
    common_kwargs = {
        "kernel_size": node_attributes.get("kernel_shape", weights.shape[2:]),
        "stride": node_attributes.get("strides", 1),
        "dilation": node_attributes.get("dilations", 1),
        "groups": node_attributes.get("group", 1),
        "padding": padding,
        "bias": bias is not None,
    }

    if op_type == "Conv":
        special_kwargs = {
            "out_channels": weights.shape[0],
            "in_channels": weights.shape[1] * common_kwargs["groups"],
        }
    elif op_type == "ConvTranspose":
        if input_padding_module is not None:
            raise NotImplementedError("ConvTranspose with non symmetrical padding is not implemented.")

        output_padding = node_attributes.get("output_padding", [0] * spatial_rank)
        special_kwargs = {
            "out_channels": weights.shape[1] * common_kwargs["groups"],
            "in_channels": weights.shape[0],
            "output_padding": output_padding,
        }
    else:
        raise ValueError(f'Got unknown op_type "{op_type}"')

    if padding == "SAME_UPPER":
        torch_module = SameUpperConv(
            **special_kwargs,
            **common_kwargs,
        )
    else:
        torch_module = conv_class(
            **common_kwargs,
            **special_kwargs,
        )
    with torch.no_grad():
        torch_module.weight.data = weights
        if bias is not None:
            torch_module.bias.data = bias

    if input_padding_module is not None:
        torch_module = nn.Sequential(input_padding_module, torch_module)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )
