# Tutorial: Qwen2-VL Implementation from Scratch <br> 从零开始实现 Qwen2-VL (MindSpore Version)
<!-- TODO: separate doc for CN ver. -->
[中文教程](README_CN.md)  &nbsp;&nbsp;|&nbsp;&nbsp; English Tutorial

> This tutorial aims to re-implement Qwen2-VL based on [MindSpore](https://gitee.com/mindspore/mindspore) and [MindONE](https://github.com/mindspore-lab/mindone).
<br> 基于 [MindSpore](https://gitee.com/mindspore/mindspore) and [MindONE](https://github.com/mindspore-lab/mindone) 实现Qwen2-VL。

**Introduction:** Qwen2-VL is an advanced version of the [Qwen-VL](https://github.com/QwenLM/Qwen-VL) model, a large visual language model (LVLM). Key improvements include enhanced image comprehension, advanced video understanding, integrated visual agent functionality, and expanded multilingual support.
<br>
The model architecture has been optimized for handling arbitrary image resolutions through [Naive Dynamic Resolution](#naive-dynamic-resolution) support and utilizes [Multimodal Rotary Position Embedding (M-ROPE)](#multimodal-rotary-position-embedding-m-rope) to effectively process both 1D textual and multi-dimensional visual data. This updated model demonstrates competitive performance against leading AI systems like GPT-4o and Claude 3.5 Sonnet in vision-related tasks and ranks highly among open-source models in text capabilities. These advancements make Qwen2-VL a versatile tool for various applications requiring robust multimodal processing and reasoning abilities.

## 1. Framework Overview 流程概览
Qwen2-VL adapted [ViT](https://github.com/google-research/vision_transformer#vision-transformer)'s encoder as Vision Encoder, and LLM [Qwen2](https://github.com/QwenLM/Qwen2)'s decoder as Decoder.

### Framework Overview 整体流程图

<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/qwen2_vl.jpg" alt="Qwen2-VL Framework" width="800"/>

_Qwen2-VL architecture. Taken from [original paper](https://arxiv.org/abs/2409.12191)._
<br>Overall flow: Feeding multimodal inputs (vision and text) with M-ROPE into ViT visual encoder, LLM Qwen2 decode encoded input tokens and return textual reponses.

### Tasks 

## Model Architecture and Modules
### 1. Model Architecture

### 2. Visual Encoder
#### Revisit Vision Transformer (ViT)
![ViT architecture](https://github.com/google-research/vision_transformer/raw/main/vit_figure.png)
<br> _ViT architecture. Taken from [original paper](https://arxiv.org/abs/2010.11929)._

```python
```

#### Qwen2-VL ViT
Architecture

```python
```

### Naive Dynamic Resolution
Qwen2-VL can handle arbitrary image resolutions, mapping them into a dynamic number of visual tokens, offering a more human-like visual processing experience. 
```python
```

### Multimodal Rotary Position Embedding (M-ROPE)
Decomposes positional embedding into parts to capture 1D textual, 2D visual, and 3D video positional information, enhancing its multimodal processing capabilities.

![M-ROPE](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/mrope.png)
#### Revisit: Rotary Position Embedding (1D-ROPE)
```python
```

#### M-ROPE
```python
```


### 3. LM Decoder
