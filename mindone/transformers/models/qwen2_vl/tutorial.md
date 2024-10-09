# Tutorial: Qwen2-VL Implementation from Scratch 从零开始实现Qwen2-VL (MindSpore Version)
<!-- TODO: separate doc for CN ver. -->
[中文教程](README_CN.md)  &nbsp;&nbsp;|&nbsp;&nbsp; English Tutorial

> This tutorial aims to re-implement Qwen2-VL based on [MindSpore]() and [MindONE]().
<br> 基于 [MindSpore]() and [MindONE]() 实现Qwen2-VL。


## 1. Framework Overview 流程概览
Qwen2-VL adapted [ViT]() as Vision Encoder, and LLM [Qwen2]() as Decoder.

![整体流程图 Framework Overview](./imgs/img1.png)

### Tasks
## Model Architecture and Modules
### 1. Vision Transformer (ViT)
### Naive Dynamic Resolution: 
Qwen2-VL can handle arbitrary image resolutions, mapping them into a dynamic number of visual tokens, offering a more human-like visual processing experience. 
### Multimodal Rotary Position Embedding (M-ROPE): 
Decomposes positional embedding into parts to capture 1D textual, 2D visual, and 3D video positional information, enhancing its multimodal processing capabilities.
#### Revisit: 1-D Rotary Position Embedding (ROPE)
#### Multimodal Rotary Position Embedding (M-ROPE)


### Qwen2
