import types
import torch.nn.functional as F

from typing import Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.attention_processor import Attention

from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
from diffusers.models.transformers.transformer_ltx import apply_rotary_emb

class LTXI2AVPipeline(LTXPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_layers(self, file_path="./unet_info.txt"):
        layers = []
        with open(file_path, "w") as f:
            for name, module in self.transformer.named_modules():
                if "attn1" in name and "to" not in name and "add" not in name and "norm" not in name:
                    f.write(f"{name}\n")
                    layers.append((name, module))
        return layers        

    # TODO: right now just for inference. May need to adjust for training
    @torch.no_grad()
    def __call__(
        self,
        #I2V
        image: PipelineImageInput = None,
        prompt_vid: Union[str, List[str]] = None,
        negative_prompt_vid: Optional[Union[str, List[str]]] = None,

        vid_height: int = 512,
        vid_width: int = 704,
        vid_num_frames: int = 161,
        vid_frame_rate: int = 25,
        vid_output_type: Optional[str] = "pil",

        do_rescaling: Optional[bool] = False,
        stg_mode: Optional[str] = "STG-R",

        #T2A
        prompt_aud: Union[str, List[str]] = None,
        negative_prompt_aud: Optional[Union[str, List[str]]] = None,
        stg_applied_layers_idx: Optional[List[int]] = [35],
        stg_scale: Optional[float] = 1.0,

        aud_mel_bins: int = 256,
        aud_time_bins: int = 512,
        aud_num_frames: int = 1,
        aud_output_type: Optional[str] = "wave", #latent or wave

        #common
        num_inference_steps: int = 50,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

        #maybe important later
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        latents: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        layers = self.extract_layers()

        # TODO: add check_inputs function 

        self._stg_scale = stg_scale
        self._guidance_scale = guidance_scale
        self._interrupt = False

        if self.do_spatio_temporal_guidance:
            if stg_mode == "STG-A":
                layers = self.extract_layers()
                replace_processor = STGLTXVideoAttentionProcessor2_0()
                self.replace_layer_processor(layers, replace_processor, stg_applied_layers_idx)
            elif stg_mode == "STG-R":
                raise NotImplementedError("STG-R is not implemented yet.")

        # 2. Define call parameters
        if prompt_vid is not None and isinstance(prompt_vid, str):
            batch_size = 1
        elif prompt_vid is not None and isinstance(prompt_vid, list):
            batch_size = len(prompt_vid)
        else:
            raise ValueError("Prompt must be a string or a list of strings.")

        device = self._execution_device

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare latent variables
        if latents is None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image = image.to(device=device, dtype=prompt_embeds.dtype)

        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])
        elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask, conditioning_mask])

        # 5. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare micro-conditions
        latent_frame_rate = frame_rate / self.vae_temporal_compression_ratio
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 3)
                    
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()
   
                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    timestep, _ = timestep.chunk(2)
                elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) \
                        + self._stg_scale * (noise_pred_text - noise_pred_perturb)
                    timestep, _, _ = timestep.chunk(3)
                    
                if do_rescaling:
                    rescaling_scale = 0.7
                    factor = noise_pred_text.std() / noise_pred.std()
                    factor = rescaling_scale * factor + (1 - rescaling_scale)
                    noise_pred = noise_pred * factor

                # compute the previous noisy sample x_t -> x_t-1
                noise_pred = self._unpack_latents(
                    noise_pred,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
                latents = self._unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )

                noise_pred = noise_pred[:, :, 1:]
                noise_latents = latents[:, :, 1:]
                pred_latents = self.scheduler.step(noise_pred, t, noise_latents, return_dict=False)[0]

                latents = torch.cat([latents[:, :, :1], pred_latents], dim=2)
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            video = latents
        else:
            latents = self._unpack_latents(
                latents,
                latent_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)

    @torch.no_grad()
    def forward(self, latents, **kwargs):
        """
        Override the forward function to apply different LoRA adapters to different parts of the latents.
        """

        # Split concatenated latents
        F1_size = latents.shape[0] // 2  # Assuming F1 and F2 have equal size
        F1_latents, F2_latents = latents[:F1_size], latents[F1_size:]

        # Apply lora_f1 to F1
        self.set_adapters("lora_f1", LORA_WEIGHT)
        output_F1 = super().forward(F1_latents, **kwargs)  # Process F1

        # Apply lora_f2 to F2
        self.set_adapters("lora_f2", LORA_WEIGHT)
        output_F2 = super().forward(F2_latents, **kwargs)  # Process F2

        # Merge F1 and F2 outputs (depending on the model’s expected format)
        merged_output = self.merge_outputs(output_F1, output_F2)

        return merged_output    
    

class STGLTXVideoAttentionProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LTX model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "LTXVideoAttentionProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        hidden_states_uncond, hidden_states_text, hidden_states_perturb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_text])
        
        emb_sin, emb_cos = image_rotary_emb
        emb_sin_uncond, emb_sin_text, emb_sin_perturb = emb_sin.chunk(3)
        emb_cos_uncond, emb_cos_text, emb_cos_perturb = emb_cos.chunk(3)
        emb_sin_org = torch.cat([emb_sin_uncond, emb_sin_text])
        emb_cos_org = torch.cat([emb_cos_uncond, emb_cos_text])
        
        image_rotary_emb_org = (emb_sin_org, emb_cos_org)
        image_rotary_emb_perturb = (emb_sin_perturb, emb_cos_perturb)
        
        #----------------Original Path----------------#
        assert encoder_hidden_states is None
        batch_size, sequence_length, _ = (
            hidden_states_org.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states_org = hidden_states_org

        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(encoder_hidden_states_org)
        value_org = attn.to_v(encoder_hidden_states_org)

        query_org = attn.norm_q(query_org)
        key_org = attn.norm_k(key_org)

        if image_rotary_emb is not None:
            query_org = apply_rotary_emb(query_org, image_rotary_emb_org)
            key_org = apply_rotary_emb(key_org, image_rotary_emb_org)

        query_org = query_org.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key_org = key_org.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value_org = value_org.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.transpose(1, 2).flatten(2, 3)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        hidden_states_org = attn.to_out[0](hidden_states_org)
        hidden_states_org = attn.to_out[1](hidden_states_org)
        #----------------------------------------------#
        #--------------Perturbation Path---------------#
        batch_size, sequence_length, _ = hidden_states_perturb.shape 

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states_perturb = hidden_states_perturb

        query_perturb = attn.to_q(hidden_states_perturb)
        key_perturb = attn.to_k(encoder_hidden_states_perturb)
        value_perturb = attn.to_v(encoder_hidden_states_perturb)

        query_perturb = attn.norm_q(query_perturb)
        key_perturb = attn.norm_k(key_perturb)

        if image_rotary_emb is not None:
            query_perturb = apply_rotary_emb(query_perturb, image_rotary_emb_perturb)
            key_perturb = apply_rotary_emb(key_perturb, image_rotary_emb_perturb)

        query_perturb = query_perturb.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key_perturb = key_perturb.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value_perturb = value_perturb.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states_perturb = value_perturb
        
        hidden_states_perturb = hidden_states_perturb.transpose(1, 2).flatten(2, 3)
        hidden_states_perturb = hidden_states_perturb.to(query_perturb.dtype)

        hidden_states_perturb = attn.to_out[0](hidden_states_perturb)
        hidden_states_perturb = attn.to_out[1](hidden_states_perturb)
        #----------------------------------------------#
        
        hidden_states = torch.cat([hidden_states_org, hidden_states_perturb], dim=0)
        
        return hidden_states