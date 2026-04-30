import os
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers import FluxControlNetPipeline
from registry.trainer_registry import TrainerRegistry
import torch.nn.functional as F
from utils.validation import flux1control_validation

@TrainerRegistry.register('flux1_controlnet')
class Flux1ControlNetTrainer:
    def __init__(self, accelerator: Accelerator, config, logger=None):
        self.accelerator = accelerator
        self.config = config
        self.global_step = 0
        self.logger = logger
        
    def train(self, train_dataloader, model_wrapper, optimizer, lr_scheduler, **kwargs):
        transformer = model_wrapper.transformer
        vae = model_wrapper.vae
        noise_scheduler = model_wrapper.scheduler
        controlnet = model_wrapper.controlnet
        controlnet.train()
        progress_bar = tqdm(range(self.config.training.max_train_steps), disable=not self.accelerator.is_local_main_process)

        while self.global_step < self.config.training.max_train_steps:  
            for _, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate([controlnet]):

                    loss = self._train_step(
                        batch, transformer, vae,controlnet,noise_scheduler
                    )
                    self.accelerator.backward(loss)
                    # 梯度累积
                    if self.accelerator.sync_gradients:
                        # 梯度裁剪
                        self.accelerator.clip_grad_norm_(controlnet.parameters(), self.config.training.max_grad_norm)
                        # 权重更新
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        self.global_step += 1
                        self._after_step(
                            loss=loss,
                            lr_scheduler=lr_scheduler,
                            controlnet=controlnet,
                            progress_bar=progress_bar
                        )
        progress_bar.close()
        self._save_final_checkpoint(controlnet)
        self.logger.info("Training finished.")

    def _after_step(self, loss, lr_scheduler, controlnet, progress_bar):
        lr = lr_scheduler.get_last_lr()[0]
        loss_value = loss.item()
        progress_bar.set_postfix({
            "loss": f"{loss_value:.4f}",
            "lr": f"{lr:.2e}"
        })
        progress_bar.update(1)
        self.logger.log_metrics(
            {"loss": loss_value, "lr": lr},
            step=self.global_step
        )
        if self.global_step % 20 == 0:
            self.logger.info(
                f"Step {self.global_step}/{self.config.training.max_train_steps} | "
                f"loss={loss_value:.4f} | lr={lr:.2e}"
            )
            self.logger.plot_curves()
        if self.global_step % self.config.training.checkpointing_steps == 0:
            self._save_checkpoint(controlnet)
        if self.global_step % self.config.validation.validation_steps == 0:
            flux1control_validation(
                self.config,
                controlnet,
                self.accelerator,
                self.global_step
            )

    def _train_step(self, batch, transformer, vae, controlnet, noise_scheduler):
        device = self.accelerator.device

        pixel_values = batch["pixel_values"].to(device=device,dtype=vae.dtype)
        pixel_latents_tmp = vae.encode(pixel_values).latent_dist.sample()
        pixel_latents_tmp = (pixel_latents_tmp - vae.config.shift_factor) * vae.config.scaling_factor
        pixel_latents = FluxControlNetPipeline._pack_latents(
            pixel_latents_tmp,
            pixel_values.shape[0],
            pixel_latents_tmp.shape[1],
            pixel_latents_tmp.shape[2],
            pixel_latents_tmp.shape[3],
        )
        
        control_values = batch["conditioning_pixel_values"].to(device=device,dtype=vae.dtype)
        control_latents = vae.encode(control_values).latent_dist.sample()
        control_latents = (control_latents - vae.config.shift_factor) * vae.config.scaling_factor
        control_image = FluxControlNetPipeline._pack_latents(
            control_latents,
            control_values.shape[0],
            control_latents.shape[1],
            control_latents.shape[2],
            control_latents.shape[3],
        )

        latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
            batch_size=pixel_latents_tmp.shape[0],
            height=pixel_latents_tmp.shape[2] // 2,
            width=pixel_latents_tmp.shape[3] // 2,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )

        bsz = pixel_latents.shape[0]
        noise = torch.randn_like(pixel_latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.config.training.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.config.training.logit_mean,
            logit_std=self.config.training.logit_std,
            mode_scale=self.config.training.mode_scale,
        )

        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = self._get_sigmas(timesteps, noise_scheduler,n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

        # handle guidance
        if transformer.config.guidance_embeds:
            guidance_vec = torch.full(
                (noisy_model_input.shape[0],),
                self.config.training.guidance_scale,
                device=noisy_model_input.device
            )
        else:
            guidance_vec = None

        controlnet_block_samples, controlnet_single_block_samples = controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=control_image,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch["unet_added_conditions"]["pooled_prompt_embeds"].to(dtype=controlnet.dtype),
                    encoder_hidden_states=batch["prompt_ids"].to(dtype=controlnet.dtype),
                    txt_ids=batch["unet_added_conditions"]["time_ids"][0].to(dtype=controlnet.dtype),
                    img_ids=latent_image_ids,
                    return_dict=False,
                )
        noise_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch["unet_added_conditions"]["pooled_prompt_embeds"].to(dtype=transformer.dtype),
                    encoder_hidden_states=batch["prompt_ids"].to(dtype=transformer.dtype),
                    controlnet_block_samples=[sample.to(dtype=transformer.dtype) for sample in controlnet_block_samples]
                    if controlnet_block_samples is not None
                    else None,
                    controlnet_single_block_samples=[
                        sample.to(dtype=transformer.dtype) for sample in controlnet_single_block_samples
                    ]
                    if controlnet_single_block_samples is not None
                    else None,
                    txt_ids=batch["unet_added_conditions"]["time_ids"][0].to(dtype=transformer.dtype),
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
        loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")
        return loss

    def _get_sigmas(self, timesteps, noise_scheduler,n_dim=4, device='cuda',dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def _save_checkpoint(self, transformer):
        if self.accelerator.is_main_process:
            self._save_controlnet(transformer, f"checkpoint-{self.global_step}")

    def _save_final_checkpoint(self, transformer):
        if self.accelerator.is_main_process:
            self._save_controlnet(transformer, "final")
    
    def _save_controlnet(self, controlnet, name):
        save_path = os.path.join(self.config.training.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        controlnet.save_pretrained(save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")


