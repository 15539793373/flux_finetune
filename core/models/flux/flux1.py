import torch
from diffusers import FluxPipeline
from registry.model_registry import ModelRegistry


@ModelRegistry.register('flux1')
class FluxModel:
    def __init__(self, config, dtype=torch.bfloat16, device="cuda"):

        self.config = config
        self.device = device
        self.dtype = dtype
        self.pipe = FluxPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=dtype
        ).to("cpu")
        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.text_encoder_one = self.pipe.text_encoder
        self.text_encoder_two = self.pipe.text_encoder_2
        self.scheduler = self.pipe.scheduler

    def to(self, device):
        self.device = device
        self.vae.to(device)
        self.transformer.to(device)
        return self

    def set_trainable(self, trainable=True):
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)

    @torch.no_grad()
    def _encode(self, prompt):
        if next(self.text_encoder_one.parameters()).device.type != self.device:
            self.text_encoder_one = self.text_encoder_one.to(self.device).eval()
            self.text_encoder_two = self.text_encoder_two.to(self.device).eval()
            
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            max_sequence_length=self.config.model.max_sequence_length,
        )
        return {
            "prompt_embeds":prompt_embeds,
            "pooled_prompt_embeds":pooled_prompt_embeds,
            "text_ids": text_ids
                   }

    def unload_text_encoder(self):
        if self.text_encoder_one and self.text_encoder_two:
            self.text_encoder_one.to("cpu")
            self.text_encoder_two.to("cpu")
            torch.cuda.empty_cache()