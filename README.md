# Tiny Inference Engine

A lightweight CPU inference engine for ResNet and similar computer vision models. Written in pure C++ using
Eigen with BLAS backend for fast matrix operations.

![image](doc.png)

## Features

* Supports all operations required for models like ResNet or VggNet
* Use pretrained TorchScript models: Python script converts to custom format
* RGB images: reads png and jpg images (using stb_image.h)
* Fuses certain operations
* Optimizes number of threads for specific model and machine
* Easy to read & hack: perfect to understand how inference works under the hood

## Performance

* Compiles in a few seconds
* Binary <1MB
* Runs ResNet18 in 35ms (vs Torch in 20ms) on Intel i5-10400 @ 2.90GHz

## Quick start

The demo runs ResNet18 on ImageNet images. Make sure to run all commands from the root level of the repository.

### Export pretrained ResNet18

Convert the pretrained TorchVision model to the custom model format. The model gets stored in the folder `model/`.

* First make sure the python dependencies are installed: ```pip install -r requirements.txt```
* Then run the script: ```python export_resnet.py```

### Build demo

Build the binary from the C++ code. You need [Eigen](https://libeigen.gitlab.io/) and BLAS installed.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Run demo

Two ImageNet samples are provided. Run the binary (```build/tiny_inference_engine```) using the exported model (in
```model/```) and the sample images (in ```data/```):

```build/tiny_inference_engine model/ data/ false```

Expected output:

```
OP=Identity|OUT=x|ARG=__INPUT__ [DEL]
OP=Conv|OUT=getitem|ARG=x conv1.weight conv1.weight_bias 3 2 
OP=Relu|OUT=relu|ARG=getitem
...
OP=Identity|OUT=__OUTPUT__|ARG=linear [DEL]
Model loading: 24ms
--------------------
Warmup run, inference: class=308
Measuring inference time (output delayed)
--------------------
Sample: fly_224x224
Predicted: class=308 label=fly
Prob: 0.670702
--------------------
...
--------------------
Sample: goldfish_224x224
Predicted: class=1 label=goldfish
Prob: 0.915123
--------------------
Runtime per sample: 35ms
```

To optimize the number of threads for the specific model and machine once run the following command
with last parameter set to `true`:
```build/tiny_inference_engine model/ data/ true```

This stores the optimal value in the model directory that will be used in the next runs (put flag back to `false`).

### Run tests

Tests only partially cover the code right now.
The goal is to at least fully cover the conv2d and maxpool2d implementations to ensure proper functioning.

After building the demo the tests can be run with this command:
```ctest --test-dir build --output-on-failure```

## Technical Insights

### How it works

* A pretrained TorchVision model is converted to ONNX (with static shapes)
* The ONNX model is exported to the graph structure (model.txt) and the parameters (*.bin)
* Topological sorting computes the "flat" sequence of operations (model.txt)
* Certain pairs of operations get fused while exporting (e.g. Conv+Relu)
* The graph structure file and parameter files are loaded in C++ and internally the model graph is recreated with the
  operations as nodes
* All intermediate tensors (including input and output) are stored in a map
* Inference simply means applying one operation at a time, reading its input, and writing its output tensor
* The Model class puts a wrapper around all of this, takes the input tensor, and returns the output tensor
* Thread optimization is done by running the model with different configurations and storing the best one

### Performance bottlenecks

The biggest bottleneck was the Conv2d operation. The naive implementation was slow due bad memory access patterns
(jumping around in memory - a lot of cache misses). Here some of the changes that helped the most:

* Improving access patterns and avoiding recomputing array indices (using raw data pointers instead)
* Sharding (splitting work across multiple threads) into groups of channels
* Switching to a matrix multiplication based Conv2d implementation
* Letting Eigen deal with matrix multiplication
* Use BLAS as an Eigen backend
* Find optimal number of threads (one value for all layers gave best results)

Overall, this brought ResNet18 runtime down from 300ms to 35ms per sample. Also, other tweaks helped, but overall Conv2d
was the biggest bottleneck.

## Benchmark

Comparison of this inference engine with Torch.
In both cases the models are warmed up before measuring the runtime.
CPU: Intel i5-10400 @ 2.90GHz.

| Model    | Tiny Inf Eng [ms] | Torch [ms] |
|:---------|:-----------------:|:----------:|
| ResNet18 |        35         |     20     |
| ResNet34 |        60         |     40     |
| ResNet50 |        80         |     50     |
