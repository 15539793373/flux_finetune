from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel


def setup_controlnet(model_wrapper, config,logger=None):

    transformer = model_wrapper.transformer
    if config.model.model_name == "flux1":
        if config.controlnet.pretrained_controlnet_path:
            logger.info("Loading existing controlnet weights")
            controlnet = FluxControlNetModel.from_pretrained(config.controlnet.pretrained_controlnet_path)
        else:
            logger.info("Initializing controlnet weights from transformer")
            controlnet = FluxControlNetModel.from_transformer(
                transformer,
                attention_head_dim=transformer.config["attention_head_dim"],
                num_attention_heads=transformer.config["num_attention_heads"],
                num_layers=config.controlnet.num_double_layers,
                num_single_layers=config.controlnet.num_single_layers,
            )
    else:
        pass

    model_wrapper.controlnet = controlnet


    # if logger is not None:
    #     logger.info(f"Train lora modules: {target_modules}")
    #     logger.info(f"rank: {config.lora.rank}, alpha: {config.lora.alpha}, dropout: {config.lora.dropout}")
    #     logger.info(f"Trainable parameters: {trainable}, Total: {total}, Ratio: {trainable/total:.6f}")