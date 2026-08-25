"""Exports pretrained ResNet18 model from TorchVision to custom format of tiny inference engine."""

import struct
from graphlib import TopologicalSorter
from pathlib import Path

import numpy as np
import onnx
import torch
from torchvision.models import resnet18, ResNet18_Weights
from typing import Any


def torch_to_onnx(model_dir: Path) -> None:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    # export model to ONNX
    with torch.no_grad():
        onnx_program = torch.onnx.export(model, torch.zeros([1, 3, 224, 224]), dynamo=True)
        onnx_program.save(model_dir / "model.onnx")

    with open(model_dir / "label.txt", "w") as f:
        f.write(",".join(weights.meta["categories"]))


def get_attribute_args(node: Any) -> list:
    res = []
    if node.op_type == "Conv":
        for n in ["pads", "strides"]:
            for a in node.attribute:
                if a.name == n:
                    res.append(str(a.ints[0]))
    if node.op_type == "MaxPool":
        for n in ["kernel_shape", "pads", "strides"]:
            for a in node.attribute:
                if a.name == n:
                    res.append(str(a.ints[0]))
    return res


def save_tensor(fn: Path, t: np.ndarray) -> None:
    with open(fn, "wb") as f:
        # num dimensions
        f.write(struct.pack('i', len(t.shape)))
        # shape
        f.write(struct.pack('i' * len(t.shape), *t.shape))
        # data
        arr = t.flatten().tolist()
        f.write(struct.pack('f' * len(arr), *arr))


def export(model_dir: Path) -> None:
    print("Exporting", model_dir)
    onnx_model = onnx.load(model_dir / "model.onnx")

    # weights
    weight_names = set()
    for node in onnx_model.graph.initializer:
        save_tensor(model_dir / f"{node.name}.bin", onnx.numpy_helper.to_array(node))
        weight_names.add(node.name)

    if len(onnx_model.graph.input) != 1 or len(onnx_model.graph.output) != 1:
        raise Exception("Not supported (single input/output required)")

    model_def = {onnx_model.graph.input[0].name: f"{onnx_model.graph.input[0].name:}=Identity(__INPUT__)",
                 "__OUTPUT__": f"__OUTPUT__=Identity({onnx_model.graph.output[0].name})"}

    ts = TopologicalSorter()
    ts.add("__OUTPUT__", onnx_model.graph.output[0].name)
    for node in onnx_model.graph.node:
        if len(node.output) != 1:
            raise Exception("Not supported (single node output required)")
        output_node = node.output[0]
        input_nodes = [e for e in node.input if e not in weight_names]
        ts.add(output_node, *input_nodes)
        args = list(node.input) + get_attribute_args(node)
        model_def[output_node] = f"{output_node}={node.op_type}({','.join(args)})"

    with open(model_dir / "model.txt", "w") as f:
        for var_name in ts.static_order():
            print(model_def[var_name])
            f.write(model_def[var_name])
            f.write("\n")
    print()


def main():
    # load model
    model_dir = Path(__file__).parent / "model"
    torch_to_onnx(model_dir)
    export(model_dir)


if __name__ == "__main__":
    main()
