# accelerate launch --config_file ./configs/uncompiled_2.yaml ltx_train.py 
# accelerate launch --config_file ./configs/deepspeed.yaml ltx_train.py 

import os, random, math
from pathlib import Path
from typing import Any, Dict
from datetime import timedelta
import argparse
import json
# ----------------------------------------------------
import torch
import matplotlib.pyplot as plt
import matplotlib
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.use('Agg')
from yaml import load, dump, Loader, Dumper
import numpy as np
# ----------------------------------------------------
import diffusers
import transformers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import export_to_video, load_image, load_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
from transformers import Conv1D
# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
    gather_object,
)
# ----------------------------------------------------
from dataset import MixedBatchSampler, PrecomputedDataset, MultiDatasetWraper
from ltx_video_lora import *
# ----------------------------------------------------
from utils.file_utils import find_files, delete_files, string_to_filename
from utils.optimizer_utils import get_optimizer, gradient_norm
from utils.memory_utils import get_memory_statistics, free_memory, make_contiguous
from utils.torch_utils import unwrap_model, align_device_and_dtype
import logging

import warnings
warnings.filterwarnings("ignore")

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
logger = get_logger("ltxtrainer")
logger.setLevel(LOG_LEVEL)


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None


class Trainer:
    def __init__(self, config_file) -> None:
        cd = load(open(config_file, "r"), Loader=Loader)
        cd.setdefault("precomputed_datasets", None)
        cd.setdefault("train_steps", None)
        cd.setdefault("logging_dir", "logs")
        cd.setdefault("report_to", "none")
        cd.setdefault("dataset_file", None)
        cd.setdefault("pin_memory", True)
        cd.setdefault("allow_tf32", True)
        cd.setdefault("scale_lr", True)
        cd.setdefault("train_type", "lora") # or full
        cd.setdefault("is_i2v", False) # t2v by default
        cd.setdefault("optimizer_8bit", True)
        cd.setdefault("optimizer_torchao", False)
        cd.setdefault("caption_dropout_technique", "phrase")
        cd.setdefault("noise_to_first_frame", 0.0)
        # ----------------- optimizer params --------
        cd.setdefault("optimizer" "adamw")
        cd.setdefault("lr", float(1e-4))
        cd.setdefault("scale_lr",  False)
        cd.setdefault("lr_scheduler", "constant_with_warmup")
        cd.setdefault("lr_warmup_steps", 1000)
        cd.setdefault("lr_num_cycles", 1)
        cd.setdefault("lr_power",  1.0)
        cd.setdefault("beta1",  0.9)
        cd.setdefault("beta2",  0.95)
        cd.setdefault("beta3",  0.999)
        cd.setdefault("weight_decay",  0.0001)
        cd.setdefault("epsilon",  float(1e-8))
        cd.setdefault("max_grad_norm",  1.0)
        # ---------------- Diffusion arguments
        cd.setdefault("flow_resolution_shifting", False)
        cd.setdefault("flow_base_image_seq_len", 256)
        cd.setdefault("flow_max_image_seq_len", 4096)
        cd.setdefault("flow_base_shift", 0.5)
        cd.setdefault("flow_max_shift", 1.15)
        cd.setdefault("flow_shift", 1.0)
        cd.setdefault("flow_weighting_scheme", "none")
        cd.setdefault("flow_logit_mean", 0.0)
        cd.setdefault("flow_logit_std", 1.0)
        cd.setdefault("flow_mode_scale", 1.29)
        cd.setdefault("prev_checkpoint", None)
        cd.setdefault("pretrained_model_name_or_path", "a-r-r-o-w/LTX-Video-0.9.1-diffusers")
        
        # cd.setdefault("enable_slicing", False)
        # cd.setdefault("enable_tiling", False)
        
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)
        # args.target_modules = args.target_modules.split(" ") if args.target_modules != "all-linear" else "all-linear"

        self.args = args
        self.state = State()

        # Tokenizers
        self.tokenizer = None
        # self.tokenizer_2 = None
        # self.tokenizer_3 = None
        # Text encoders
        self.text_encoder = None
        # self.text_encoder_2 = None
        # self.text_encoder_3 = None

        # Denoisers
        self.transformer = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        self.model_config = LTX_VIDEO_T2V_LORA_CONFIG
        self.losses = []

        default_prompt_file = "data/default_prompt_embedding.pth"
        if os.path.exists(default_prompt_file):
            aa = torch.load(default_prompt_file)
            self.default_prompt_embeds = aa["prompt_embeds"]
            self.default_prompt_attention_mask = aa["prompt_attention_mask"]
            print("----> load default prompt_embeds", self.default_prompt_embeds.shape)

    
    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype
        
    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)
        
    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir
    
    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.precomputed_datasets is not None:
            print("use mixed datasets", self.args.precomputed_datasets)
            list_of_datasets = [ PrecomputedDataset(data_root) for data_root in self.args.precomputed_datasets]
            dataset_indices = MultiDatasetWraper.calc_index_for_multi_datasets(list_of_datasets)

            mixed_batch_sampler = MixedBatchSampler(dataset_indices, batch_size=self.args.batch_size)

            self.dataset = MultiDatasetWraper(list_of_datasets)
            self.dataloader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_sampler=mixed_batch_sampler,
                # collate_fn=self.model_config.get("collate_fn"),
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.pin_memory,
            )
        else:
            self.dataset = PrecomputedDataset(data_dir=self.args.data_root)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.args.batch_size,
                # sampler=BucketSampler(self.dataset, batch_size=self.args.batch_size, shuffle=True),
                # collate_fn=self.model_config.get("collate_fn"),
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.pin_memory,
            )
    def prepare_models(self):
        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype
        
        # >> we use precomputation so text encoder is not needed
        # cond_models = load_condition_models()
        # tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        # self.text_encoder = text_encoder.to(device, dtype=dtype)
        
        # self.vae = load_latent_models()["vae"].to(device, dtype=dtype)

        # if self.vae is not None:
        #     if self.args.enable_slicing:
        #         self.vae.enable_slicing()
        #     if self.args.enable_tiling:
        #         self.vae.enable_tiling()
        diff_models = load_diffusion_models()
        self.transformer = diff_models["transformer"].to(device, dtype=dtype)
        self.scheduler = diff_models["scheduler"]
        self.transformer_config = self.transformer.config if self.transformer is not None else None

    @staticmethod
    def get_all_linear_names(model):
        # Create a list to store the layer names
        layer_names = []
        
        # Recursively visit all modules and submodules
        for name, module in model.named_modules():
            # Check if the module is an instance of the specified layers
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                # model name parsing 

                layer_names.append(name)
        
        return layer_names

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        if self.args.train_type == "lora":
            components_to_disable_grads = [ self.transformer ] # self.vae 
        else:
            components_to_disable_grads = []
            
        for component in components_to_disable_grads:
            if component is not None:
                component.requires_grad_(False)

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

    
        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()


        if self.args.train_type == "lora":
            target_mods = ""
            if self.args.target_modules == "all-linear":
                target_mods = self.get_all_linear_names(self.transformer)
                # target_mods = "all-linear"
            else:
                target_mods = self.args.target_modules.split(" ")

            # print("train lora layers:", target_mods)

            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                # init_lora_weights="pissa",
                init_lora_weights=True,
                target_modules=target_mods,
                lora_dropout=0.01,
            )
            self.transformer.add_adapter(transformer_lora_config)
            
        # TODO: refactor
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.state.accelerator.is_main_process:
                transformer_lora_layers_to_save = None
                
                for model in models:
                    if isinstance(
                        unwrap_model(self.state.accelerator, model),
                        type(unwrap_model(self.state.accelerator, self.transformer)),
                    ):
                        model = unwrap_model(self.state.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.model_config["pipeline_cls"].save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            transformer_ = self.model_config["pipeline_cls"].from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.set_adapters(transformer_lora_config)

            lora_state_dict = self.model_config["pipeline_cls"].lora_state_dict(input_dir)

            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if self.args.mixed_precision == "fp16":
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params([transformer_])

        self.state.accelerator.register_save_state_pre_hook(save_model_hook)
        self.state.accelerator.register_load_state_pre_hook(load_model_hook) 


        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        
        if self.args.prev_checkpoint is not None:
            print("------ load prev lora weights ------")
            lora_state_dict = self.model_config["pipeline_cls"].lora_state_dict(self.args.prev_checkpoint)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(self.transformer, transformer_state_dict, adapter_name="default")
            

    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
        # if self.args.mixed_precision == "fp16" or self.args.mixed_precision == "bf16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([self.transformer], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        transformer_parameters_with_lr = {
            "params": transformer_lora_parameters,
            "lr": self.state.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in transformer_lora_parameters)

        # TODO(aryan): add deepspeed support
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
    def prepare_for_training(self):
        self.transformer, self.optimizer, self.dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.transformer, self.optimizer, self.dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        # num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.args.gradient_accumulation_steps)
        # if self.state.overwrote_max_train_steps:
        #     self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        # self.state.train_epochs = math.ceil(self.state.train_steps / num_update_steps_per_epoch)
        
    def prepare_trackers(self):
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "ltx_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)
        
    def train(self):
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        
        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        # loss spikes
        anomalies = []

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            self.transformer.train()

            for step, batch in enumerate(self.dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate([ self.transformer ]):
                    latents, first_frame, prompt_embeds, prompt_attention_mask, caption, meta_info = batch
                    
                    latents = latents.to(accelerator.device, dtype=weight_dtype).contiguous()
                    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype).contiguous()
                    prompt_attention_mask = prompt_attention_mask.to(accelerator.device, dtype=weight_dtype)
                    batch_size = latents.shape[0]
                    
                    # 对于precompute的数据，如果直接把emebedding变成0非常容易让网络崩溃
                    # 更好的方法是，在生成precompute的时候，生成多个text embedd,训练时dataset随机返回带有破损的text embedd
                    # @TODO: 把以下逻辑放到dataset.py
                    if self.args.caption_dropout_technique == "zero":
                        if random.random() < self.args.caption_dropout_p:
                            # text_conditions["prompt_embeds"].fill_(0)
                            prompt_embeds.fill_(0)
                            prompt_attention_mask.fill_(False)

                            # if "pooled_prompt_embeds" in text_conditions:
                            #     text_conditions["pooled_prompt_embeds"].fill_(0)
                    # randomly use short phrash embeddings
                    elif self.args.caption_dropout_technique == "phrase":
                        if random.random() < self.args.caption_dropout_p:
                            prompt_embeds = self.default_prompt_embeds.repeat(prompt_embeds.size(0), 1, 1).to(accelerator.device, dtype=weight_dtype).contiguous()
                            prompt_attention_mask = self.default_prompt_attention_mask.repeat(prompt_embeds.size(0), 1).to(accelerator.device, dtype=weight_dtype).contiguous()

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    indices = (weights * self.scheduler.config.num_train_timesteps).long()
                    sigmas = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()

                    # print("timesteps", timesteps.shape, timesteps)
                    # print("first_frame should be (batch, packed, 128)", first_frame.shape,  meta_info["num_frames"][0], meta_info["height"][0], meta_info["width"][0])
                    if self.args.is_i2v:
                        noise, conditioning_mask = gen_noise_from_first_frame_latent(first_frame, meta_info["num_frames"][0], meta_info["height"][0], meta_info["width"][0], noise_to_first_frame=self.args.noise_to_first_frame)
                        # do not denoise first frame
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
                        # print("maske applied timesteps", timesteps.shape, timesteps)
                        # print("sigmas", sigmas.shape)
                        # print("conditioning_mask", conditioning_mask.shape)

                    else:
                        noise = torch.randn(
                            latents.shape,
                            generator=generator,
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )
                    ss= sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    # print("ss sigmas", ss.shape, latents.shape)
                    # print(ss)
                    # assert False

                    noisy_latents = (1.0 - ss) * latents + ss * noise

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    weights = compute_loss_weighting_for_sd3(
                        weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas
                    ).reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    # print("weights", weights)
                    pred = self.model_config["forward_pass"](
                        transformer=self.transformer, 
                        timesteps=timesteps, 
                        latents=latents,
                        noisy_latents=noisy_latents,
                        prompt_embeds=prompt_embeds, 
                        prompt_attention_mask=prompt_attention_mask,
                        num_frames=meta_info["num_frames"][0],
                        height=meta_info["height"][0],
                        width=meta_info["width"][0],
                    )
                    target = noise - latents
                    loss = weights.float() * (pred["latents"].float() - target.float()).pow(2)
                    if self.args.is_i2v:
                        # print("loss", loss.shape, conditioning_mask.shape)
                        loss = loss * (1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss.size(-1)))

                    # Average loss across channel dimension
                    loss = loss.mean(list(range(1, loss.ndim)))
                    # Average loss across batch dimension
                    loss = loss.mean()
                    assert torch.isnan(loss) == False, "NaN loss detected"
                    accelerator.backward(loss)
                    if self.args.train_type == "lora" and loss > 1.2 and accelerator.is_main_process:
                        anomalies.append(step)
                        print("!! warning !! gradient explosion detected! should stop")
                        if len(anomalies) > 1:
                            assert anomalies[-1] - anomalies[-2] > 2, "gradient explosion confirmed! you should restart from checkpoint"

                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(self.transformer.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # if global_step == 10 and self.accelerator.is_main_process:
                    #     self.plot_loss()

                    # Checkpointing
                    # accelerator.distributed_type == DistributedType.DEEPSPEED or 
                    if accelerator.is_main_process:
                        self.losses.append(loss.item())

                        if global_step % self.args.checkpointing_steps == 0:
                            # before saving state, check if this save would set us over the `checkpointing_limit`
                            if self.args.checkpointing_limit is not None:
                                checkpoints = find_files(self.args.output_dir, prefix="checkpoint")

                                # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.args.checkpointing_limit:
                                    num_to_remove = len(checkpoints) - self.args.checkpointing_limit + 1
                                    checkpoints_to_remove = checkpoints[0:num_to_remove]
                                    delete_files(checkpoints_to_remove)

                            logger.info(f"Checkpointing at step {global_step}")
                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            # accelerator.wait_for_everyone()
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                            # ---------------
                            self.plot_loss()

                # Maybe run validation
                should_run_validation = (
                    self.args.validation_steps is not None
                    and global_step % self.args.validation_steps == 0
                )
                if should_run_validation:
                    self.validate(global_step)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    print(">>> max train step reached")
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            # Maybe run validation
            # should_run_validation = (
            #     self.args.validation_every_n_epochs is not None
            #     and (epoch + 1) % self.args.validation_every_n_epochs == 0
            # )
            # if should_run_validation:
            #     self.validate(global_step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.transformer = unwrap_model(accelerator, self.transformer)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else torch.bfloat16
                if self.args.mixed_precision == "bf16"
                else torch.float32
            )
            self.transformer = self.transformer.to(dtype)
            transformer_lora_layers = get_peft_model_state_dict(self.transformer)

            self.model_config["pipeline_cls"].save_lora_weights(
                save_directory=self.args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
            )

        del self.transformer, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()
    
    def validate(self):
        logger.info("should validate")
        pass

    def evaluate(self):
        logger.info("Starting evaluate")
        pass

    def plot_loss(self, show_plot=False):
        plt.rcParams["figure.figsize"] = (10,5)
        # fig = plt.figure(figsize=(20, 10))

        m = len(self.dataloader)

        y = np.array(self.losses)
        x = np.arange(y.shape[0], dtype=y.dtype)
        x /= m
        plt.plot(x, np.log(y), label="train loss", alpha=.5, linewidth=.05)

        n = y.shape[0] // m
        y = y[:n * m].reshape((n, m)).mean(axis=1)
        x = np.arange(n, dtype=y.dtype) + 0.5
        plt.plot(x, np.log(y), label="train loss (epoch mean)", alpha=.8)

        plt.xlabel(f"Epoch")
        plt.ylabel("Log loss")
        plt.legend()
        
        loss_plot_path = os.path.join(self.args.output_dir, "losses.png")
        plt.savefig(loss_plot_path, bbox_inches="tight")
        print(f"plot saved at {loss_plot_path}")
        if show_plot:
            plt.show()
        plt.close()


        
# trainer = Trainer("ltx_training/configs/ltx.yaml")


def main():
    trainer = Trainer("configs/ltx.yaml")

    trainer.prepare_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_optimizer()
    trainer.prepare_for_training()
    trainer.prepare_trackers()

    # print(">>> load prev state")
    # trainer.state.accelerator.load_state("/home/eisneim/www/ml/_video_gen/ltx_training/data/checkpoint-9000")

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
  
    main()
    
  
