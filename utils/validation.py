import os
import torch
from diffusers import Flux2KleinPipeline,FluxControlNetPipeline
from diffusers.training_utils import free_memory
from PIL import Image
def flux2kelin_validation(config, transformer, accelerator,global_step):

    transformer = accelerator.unwrap_model(transformer)
    transformer.eval()
    # 创建 pipeline
    pipeline = Flux2KleinPipeline.from_pretrained(
        config.model.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # 随机生成器
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(config.validation.seed)
    )
    validation_image = Image.open(config.validation.validation_image).convert("RGB") if config.validation.validation_image else None
    with torch.autocast(accelerator.device.type):
        image = pipeline(
            image=validation_image,
            prompt=config.validation.validation_prompt,
            generator=generator,
        ).images[0]

    # 保存图片
    save_path = os.path.join(config.training.output_dir,'validation_images')
    os.makedirs(save_path, exist_ok=True)
    image.save(os.path.join(save_path, f"{global_step}.jpg"))
    
    del pipeline
    free_memory()
    transformer.train()

def flux1control_validation(config, transformer, accelerator,global_step):

    transformer = accelerator.unwrap_model(transformer)
    transformer.eval()
    # 创建 pipeline
    pipeline = FluxControlNetPipeline.from_pretrained(
        config.model.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # 随机生成器
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(config.validation.seed)
    )
    validation_image = Image.open(config.validation.validation_image).convert("RGB") if config.validation.validation_image else None
    with torch.autocast(accelerator.device.type):
        image = pipeline(
            image=validation_image,
            prompt=config.validation.validation_prompt,
            generator=generator,
        ).images[0]

    # 保存图片
    save_path = os.path.join(config.training.output_dir,'validation_images')
    os.makedirs(save_path, exist_ok=True)
    image.save(os.path.join(save_path, f"{global_step}.jpg"))
    
    del pipeline
    free_memory()
    transformer.train()