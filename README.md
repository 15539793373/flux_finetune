# Flux Finetune

基于 Diffusers 官方脚本重构的 **Flux微调项目**。本项目将原单体脚本模块化，支持配置驱动、插件注册机制，便于扩展和维护。

## 🌟 当前支持

- **Flux.2-klein-text2image-lora**
- **Flux.2-klein-image2image-lora**

## 🚧 后续计划 (TODO)

- [ ] 支持 Flux.2 Dev 全量微调 (Full Fine-tune)
- [ ] 添加 ControlNet 支持
- [ ] 添加推理模块
- [ ] 优化低显存训练策略

## Train a Lora to Flux.2-klein-text2image

## 🚀 快速开始

### 1. 环境准备

```
git clone https://github.com/yourusername/flux-finetune.git
cd flux-finetune
pip install -r requirements.txt
```

### 2. 数据集准备

现在让我们获取数据集。在这个例子中，我们将使用一些狗的图片: <https://huggingface.co/datasets/diffusers/dog-example>

我们先将其下载到本地：

```
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=dog/target_image, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

制作数据集json：

```
python utils/make_datajson.py \
-target_dir dog/target_image \
--output_file dog/train.jsonl \
--prompt_single "a photo of sks dog"
参数说明：
-target_dir: 包含训练图像的文件夹路径
--output_file: 输出的训练数据列表文件路径。
(可选) --prompt\_single: 所有图像共用的提示词。
(可选) --condition\_dir: imagetoimage包含指导图像文件夹路径。
```

### 3. 配置训练参数

```
# configs/flux2kleintext2image_lora.yaml 示例片段
model:
name: "black-forest-labs/FLUX.2-klein-base-4B" # 或您的本地路径
train:
output_dir: "./outputs/dog-lora"
resolution: 1024
data:
data_json: "dog/train.jsonl"
```

### 4. 开始训练

```
python train.py --config configs/flux2kleintext2image_lora.yaml
```

### 5.部署推理demo

```python
# demo/flux2_klenin.py

def create_pipe():
    model_dir = ""
    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
        )
    pipe.load_lora_weights('')
    pipe.set_adapters(["default_0"], adapter_weights=[0.4])
    print(pipe.get_active_adapters())
    print("all:", pipe.get_list_adapters())
    pipe.to('cuda')
    return pipe
```

# 🤝 致谢

感谢 HuggingFace Diffusers 团队提供的优秀基础架构。\
感谢 Black Forest Labs 开源 Flux 模型。\
感谢所有贡献者和使用者。

**如果本项目对您有帮助，欢迎 Star ⭐️ 支持！**
