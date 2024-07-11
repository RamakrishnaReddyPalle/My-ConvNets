# ***MobileNetV3 Exploration***

This repository is dedicated to my interest in the in-depth study of various convolutional neural networks (ConvNets), starting with MobileNetV3 (with the background of V1 and V2. The goal is to understand the architecture, implementation, and performance of these models. Future repositories will cover other significant ConvNets like EfficientNet.

## **Table of Contents**
- [Introduction](#introduction)
- [MobileNetV3](#mobilenetv3)
  - [Configurations](#configurations)
  - [Architecture](#architecture)
- [Requirements](#requirements)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

## **Introduction**
This repository is created to learn in-depth about various ConvNets. MobileNetV3 is the first model explored here. MobileNetV3 is known for its efficiency and performance in mobile and embedded vision applications. The repository includes:
- Scripts and modules to define MobileNetV3
- Example notebook to demonstrate blocks & model usage
- Configuration files for model parameters


## **MobileNetV3**
### **Architecture**
MobileNetV3 combines depthwise separable convolutions and lightweight attention modules to achieve efficient performance. The **Large** model is designed for higher accuracy, while the **Small** model is optimized for speed and low resource usage.

#### **MobileNet-V3 Block**
![monilenetv3_block](https://github.com/RamakrishnaReddyPalle/My-ConvNets/assets/137872198/22531b89-1c41-42f1-9036-ffeead1f32c7)

#### **Configurations**
MobileNetV3 comes in two configurations: **Large** and **Small**. Each configuration is tailored for different resource constraints and performance requirements.



**Parameters Table**

**Large Configuration:**

| Input        | Operator      | exp size | \#out | SE  | NL  | s  |
|--------------|---------------|----------|-------|-----|-----|----|
| 224<sup>2</sup> x 3   | conv2d       | -        | 16    | -   | HS  | 2  |
| 112<sup>2</sup> x 16  | bneck, 3x3   | 16       | 16    | -   | RE  | 1  |
| 112<sup>2</sup> x 16  | bneck, 3x3   | 64       | 24    | -   | RE  | 2  |
| 56<sup>2</sup> x 24   | bneck, 3x3   | 72       | 24    | -   | RE  | 1  |
| 56<sup>2</sup> x 24   | bneck, 5x5   | 72       | 40    | ✓   | RE  | 2  |
| 28<sup>2</sup> x 40   | bneck, 5x5   | 120      | 40    | ✓   | RE  | 1  |
| 28<sup>2</sup> x 40   | bneck, 5x5   | 120      | 40    | ✓   | RE  | 1  |
| 28<sup>2</sup> x 40   | bneck, 3x3   | 240      | 80    | -   | HS  | 2  |
| 14<sup>2</sup> x 80   | bneck, 3x3   | 200      | 80    | -   | HS  | 1  |
| 14<sup>2</sup> x 80   | bneck, 3x3   | 184      | 80    | -   | HS  | 1  |
| 14<sup>2</sup> x 80   | bneck, 3x3   | 184      | 80    | -   | HS  | 1  |
| 14<sup>2</sup> x 80   | bneck, 3x3   | 480      | 112   | ✓   | HS  | 1  |
| 14<sup>2</sup> x 112  | bneck, 3x3   | 672      | 112   | ✓   | HS  | 1  |
| 7<sup>2</sup> x 112   | bneck, 5x5   | 672      | 160   | ✓   | HS  | 2  |
| 7<sup>2</sup> x 160   | bneck, 5x5   | 960      | 160   | ✓   | HS  | 1  |
| 7<sup>2</sup> x 160   | bneck, 5x5   | 960      | 160   | ✓   | HS  | 1  |
| 7<sup>2</sup> x 160   | conv2d, 1x1  | -        | 960   | -   | HS  | 1  |
| 7<sup>2</sup> x 960   | pool, 7x7    | -        | -     | -   | -   | -  |
| 1<sup>2</sup> x 960   | conv2d, 1x1, NBN | -    | 1280  | -   | HS  | -  |
| 1<sup>2</sup> x 1280  | conv2d, 1x1, NBN | -    | k     | -   | -   | -  |

*Table 1. Specification for MobileNetV3-Large. SE denotes whether there is a Squeeze-And-Excite in that block. NL denotes the type of nonlinearity used. Here, HS denotes h-swish and RE denotes ReLU. NBN denotes no batch normalization. s denotes stride.*

**Small Configuration:**

| Input        | Operator      | exp size | \#out | SE  | NL  | s  |
|--------------|---------------|----------|-------|-----|-----|----|
| 224<sup>2</sup> x 3   | conv2d, 3x3  | -        | 16    | ✓   | HS  | 2  |
| 112<sup>2</sup> x 16  | bneck, 3x3   | 16       | 16    | ✓   | RE  | 1  |
| 56<sup>2</sup> x 16   | bneck, 3x3   | 72       | 24    | -   | RE  | 2  |
| 28<sup>2</sup> x 24   | bneck, 3x3   | 88       | 24    | -   | RE  | 1  |
| 28<sup>2</sup> x 24   | bneck, 5x5   | 96       | 40    | ✓   | HS  | 2  |
| 14<sup>2</sup> x 40   | bneck, 5x5   | 240      | 40    | ✓   | HS  | 1  |
| 14<sup>2</sup> x 40   | bneck, 5x5   | 240      | 40    | ✓   | HS  | 1  |
| 14<sup>2</sup> x 40   | bneck, 5x5   | 120      | 48    | ✓   | HS  | 1  |
| 14<sup>2</sup> x 48   | bneck, 5x5   | 144      | 48    | ✓   | HS  | 1  |
| 14<sup>2</sup> x 48   | bneck, 5x5   | 288      | 96    | ✓   | HS  | 2  |
| 7<sup>2</sup> x 96    | bneck, 5x5   | 576      | 96    | ✓   | HS  | 1  |
| 7<sup>2</sup> x 96    | bneck, 5x5   | 576      | 96    | ✓   | HS  | 1  |
| 7<sup>2</sup> x 96    | conv2d, 1x1  | -        | 576   | -   | HS  | 1  |
| 7<sup>2</sup> x 576   | pool, 7x7    | -        | -     | -   | -   | -  |
| 1<sup>2</sup> x 576   | conv2d, 1x1, NBN | -    | 1024  | -   | HS  | -  |
| 1<sup>2</sup> x 1024  | conv2d, 1x1, NBN | -    | k     | -   | -   | -  |

*Table 2. Specification for MobileNetV3-Small. See Table 1 for descriptions of the columns.*

## **Requirements**
Ensure you have the necessary libraries installed. You can use the provided `requirements.txt` file to set up your environment.

```sh
pip install -r requirements.txt
```

## **Usage**
1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/mobilenetv3-exploration.git
    cd mobilenetv3-exploration
    ```

2. **Run the Example Notebook**:
    Open and run `example.ipynb` to see the model in action with a sample input image.

3. **Model Summary**:
    Use the `main.py` and the `sample_image.jpg` script to see the model summary.

## Future Work
This repository is the beginning of a series of explorations into ConvNets. Future repositories will cover:
- EfficientNet
- ResNet
- ...
- ...
- Other significant ConvNets

## References
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [MobileNetV1 Paper](https://arxiv.org/abs/1704.04861)
- [Depth-wise Seperable Convolutions](https://arxiv.org/abs/1610.02357)
- [Residual Connections](https://paperswithcode.com/method/residual-connection)
- [Inverted Residual Connections](https://paperswithcode.com/method/inverted-residual-block)
- [Squeeze & Excitation Paper](https://arxiv.org/abs/1709.01507)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TorchVision Documentation](https://pytorch.org/vision/stable/index.html)

---
