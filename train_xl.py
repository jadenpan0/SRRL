import os
import contextlib
import datetime
import copy
import time
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, UNet2DConditionModel, DDIMInverseScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusion.ddim_with_logprob import ddim_step_with_logprob
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

import numpy as np
# from diffusion.pipeline_with_logprob import pipeline_with_logprob
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
import torch
from functools import partial
import tqdm
from PIL import Image
import tree
import json
import pickle
from utils.utils import post_processing,load_sample_stage, seed_everything
import random
from utils.utils import seed_everything
from rewards.rewards import score_fn1, score_fn2, score_fn3

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/stage_process.py", "Sampling configuration.")


logger = get_logger(__name__)

def run_sample_main():
    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids

    def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            if tokenizers is not None:
                tokenizer = tokenizers[i]
                text_input_ids = tokenize_prompt(tokenizer, prompt)
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    # basic Accelerate and logging setup
    config = FLAGS.config
    debug_idx = 0
    print(f'========== seed: {config.seed} ==========')
    # print(config.prompt)
    torch.cuda.set_device(config.dev_id)

    if config.exp_name:
        unique_id = config.exp_name
    else: 
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)

    if config.run_name:
        stage_id = config.run_name
    else: 
        stage_id = "stage"+str(os.listdir(os.path.join(config.save_path, unique_id)))
        
    save_dir = os.path.join(config.save_path, unique_id, stage_id)

    if config.resume_from:
        print("loading model. Please Wait.")
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print("load successfully!")

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir, # os.path.join(config.logdir, config.run_name), #
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )

    # load scheduler, tokenizer and models.
    ####################################
    pipeline = StableDiffusionXLPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16) # float16
    inv_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    # pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model, torch_dtype=torch.float16)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    #pipeline.tokenizer.requires_grad_(False)
    #pipeline.tokenizer_2.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    total_image_num_per_gpu = config.sample.batch_size * config.sample.num_batches_per_epoch
    
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    #pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    
    rank=4
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        pipeline.unet.add_adapter(unet_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(pipeline.unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(unwrap_model(model), type(unwrap_model(pipeline.text_encoder))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(unwrap_model(model), type(unwrap_model(pipeline.text_encoder_2))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(pipeline.unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(pipeline.text_encoder))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(pipeline.text_encoder_2))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
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
        if accelerator.mixed_precision == "fp16":
            models = [unet_]
            cast_training_params(models, dtype=torch.float32)
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Define tokenizers and text encoders
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2] if pipeline.tokenizer is not None else [pipeline.tokenizer_2]
    text_encoders = (
        [pipeline.text_encoder, pipeline.text_encoder_2] if pipeline.text_encoder is not None else [pipeline.text_encoder_2]
    )
    
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    pipeline.unet = accelerator.prepare(pipeline.unet)

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)

    # set_seed(config.seed, device_specific=True)
    seed_everything(config.seed)

    #################### SAMPLING ####################
    pipeline.unet.eval()
    samples = []
    split_steps = [config.split_step]
    split_times = [config.split_time]

    total_prompts = []
    total_samples = None
    if os.path.exists(os.path.join(save_dir, f'prompt.json')): 
        with open(os.path.join(save_dir, f'prompt.json'), 'r') as f: 
            total_prompts = json.load(f)
    if os.path.exists(os.path.join(save_dir, f'sample.pkl')): 
        with open(os.path.join(save_dir, f'sample.pkl'), 'rb') as f: 
            total_samples = pickle.load(f)
    global_idx = len(total_prompts)
    local_idx = 0

    prompt_list = []
    if len(config.prompt)==0:
        with open(config.prompt_file) as f:
            prompt_list = json.load(f)
    prompt_idx = 0
    prompt_cnt = len(prompt_list)

    for idx in tqdm(
        range(config.sample.num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        # generate prompts
        if len(config.prompt)!=0:
            prompts1 = [config.prompt for _ in range(config.sample.batch_size)] 
        elif config.prompt_random_choose:
            prompts1 = [random.choice(prompt_list) for _ in range(config.sample.batch_size)] 
        else:
            prompts1 = [prompt_list[(prompt_idx+i)%prompt_cnt] for i in range(config.sample.batch_size)] 
            prompt_idx += config.sample.batch_size
        
        prompt_embeds1,negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds=pipeline.encode_prompt(prompts1,None, accelerator.device)

        saved_add_text_embeds=pooled_prompt_embeds

        noise_latents1 = pipeline.prepare_latents(
            config.sample.batch_size, 
            pipeline.unet.config.in_channels, ## channels
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
            prompt_embeds1.dtype, 
            accelerator.device, 
            None ## generator
        )

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

        height = pipeline.default_sample_size * pipeline.vae_scale_factor
        width = pipeline.default_sample_size * pipeline.vae_scale_factor
        original_size =  (height, width)
        target_size =  (height, width)

        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            (0, 0),
            target_size,
            dtype=prompt_embeds1.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        prompt_embeds1_combine = torch.cat([negative_prompt_embeds, prompt_embeds1], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds1_combine = prompt_embeds1_combine.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.to(accelerator.device).repeat(config.sample.batch_size, 1, 1)

        pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
        inv_scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)

        @torch.no_grad()
        def prepare_resample_noise_latent(noise_latents1, resample_num, forward_guidance_scale=3.0, backward_guidance_scale=0.1):
            latents=noise_latents1
            forward_timesteps=pipeline.scheduler.timesteps
            backward_timesteps=torch.flip(forward_timesteps, [0]).to(accelerator.device)
            extra_step_kwargs={"eta": 0.0, "generator": None}
            #backward_timesteps=forward_timesteps[::-1]
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            for _ in range(resample_num):
                for i,t in enumerate(forward_timesteps):
                    latents_input = torch.cat([latents] * 2) if forward_guidance_scale else latents
                    latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                    noise_pred = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                    if forward_guidance_scale:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + forward_guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                for i,t in enumerate(backward_timesteps):
                    latents_input = torch.cat([latents] * 2) if backward_guidance_scale else latents
                    latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                    noise_pred = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                    if backward_guidance_scale:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + backward_guidance_scale * (noise_pred_text - noise_pred_uncond)
                    inv_latents = inv_scheduler.step(noise_pred, backward_timesteps[i], latents, return_dict=False)[0]
                    latents = inv_latents
            #print(latents.dtype)
            return latents.type(torch.float16)
        
        noise_latents1=prepare_resample_noise_latent(noise_latents1, config.resample_num)
            
        pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
        ts = pipeline.scheduler.timesteps
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, config.sample.eta)

        latents = [[noise_latents1]] ## [t, batch_size, channels, height, width]    latent[t]
        log_probs = [[]]

        for i, t in tqdm(
            enumerate(ts),
            desc="Timestep",
            position=3,
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):  
            # sample
            with autocast():
                with torch.no_grad():
                    if ((config.sample.num_steps-i) in split_steps):
                    #if i==0:
                        branch_num = split_steps.index(config.sample.num_steps-i)
                        branch_num = split_times[branch_num]
                        branch_num = config.split_time
                        cur_sample_num = len(latents)
                        # split the sample
                        latents = [
                            [latent for latent in latents[k//branch_num]] 
                            for k in range(cur_sample_num*branch_num)
                            ]
                        log_probs = [
                            [log_prob for log_prob in log_probs[k//branch_num]] 
                            for k in range(cur_sample_num*branch_num)
                            ]
                        
                    for k in range(len(latents)): 
                        latents_t = latents[k][i]
                        latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)

                        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                        #print(add_text_embeds.shape, add_time_ids.shape, prompt_embeds1_combine.shape)
                        noise_pred = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        latents_t_1, log_prob, latents_0 = ddim_step_with_logprob(pipeline.scheduler, noise_pred, t, latents_t, **extra_step_kwargs)
                        
                        latents[k].append(latents_t_1)
                        log_probs[k].append(log_prob)

        sample_num = len(latents)
        total_prompts.extend(prompts1*sample_num)

        for k in range(sample_num):    
            images = latents_decode(pipeline, latents[k][config.sample.num_steps], accelerator.device, prompt_embeds1.dtype)
            store_latents = torch.stack(latents[k], dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            store_log_probs = torch.stack(log_probs[k], dim=1)  # (batch_size, num_steps)
            prompt_embeds = prompt_embeds1
            current_latents = store_latents[:, :-1]
            next_latents = store_latents[:, 1:]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            samples.append(
                {
                    "prompt_embeds": prompt_embeds.cpu().detach(),
                    "timesteps": timesteps.cpu().detach(),
                    "log_probs": store_log_probs.cpu().detach(),
                    "latents": current_latents.cpu().detach(),  # each entry is the latent before timestep t
                    "next_latents": next_latents.cpu().detach(),  # each entry is the latent after timestep t
                    "images":images.cpu().detach(),
                    "add_text_embeds": add_text_embeds.cpu().detach().unsqueeze(0),
                    "add_time_ids": add_time_ids.cpu().detach(),
                    "prompt_embeds1_combine": prompt_embeds1_combine.cpu().detach().unsqueeze(0)
                    #"images":images
                }
            )

        if idx==0:
            for k,v in samples[0].items():
                print(k, v.shape)

        if (idx+1)%config.sample.save_interval ==0 or idx==(config.sample.num_batches_per_epoch-1):
            os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
            print(f'-----------{accelerator.process_index} save image start-----------')
            # print(samples)
            new_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            images = new_samples['images'][local_idx:]
            for j, image in enumerate(images):
                #pil = image
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir, f"images/{(j+global_idx):05}.png"))

            global_idx += len(images)
            local_idx += len(images)
            with open(os.path.join(save_dir, f'prompt.json'),'w') as f:
                json.dump(total_prompts, f)
            with open(os.path.join(save_dir, f'sample.pkl'), 'wb') as f:
                if total_samples is None: 
                    pickle.dump({
                        "prompt_embeds": new_samples["prompt_embeds"], 
                        "timesteps": new_samples["timesteps"], 
                        "log_probs": new_samples["log_probs"], 
                        "latents": new_samples["latents"], 
                        "next_latents": new_samples["next_latents"],
                        "add_text_embeds": new_samples["add_text_embeds"],
                        "add_time_ids": new_samples["add_time_ids"],
                        "prompt_embeds1_combine": new_samples["prompt_embeds1_combine"]
                        }, f)
                else: 
                    pickle.dump({
                        "prompt_embeds": torch.cat([total_samples["prompt_embeds"], new_samples["prompt_embeds"]]), 
                        "timesteps": torch.cat([total_samples["timesteps"], new_samples["timesteps"]]), 
                        "log_probs": torch.cat([total_samples["log_probs"], new_samples["log_probs"]]), 
                        "latents": torch.cat([total_samples["latents"], new_samples["latents"]]), 
                        "next_latents": torch.cat([total_samples["next_latents"], new_samples["next_latents"]]),
                        "add_text_embeds": torch.cat([total_samples["add_text_embeds"], new_samples["add_text_embeds"]]),
                        "add_time_ids": torch.cat([total_samples["add_time_ids"], new_samples["add_time_ids"]]),
                        "prompt_embeds1_combine": torch.cat([total_samples["prompt_embeds1_combine"], new_samples["prompt_embeds1_combine"]])
                        }, f)

def run_select_main():

    # basic Accelerate and logging setup
    config = FLAGS.config

    torch.cuda.set_device(config.dev_id)
    seed_everything(config.seed)

    unique_id = config.exp_name

    if config.run_name:
        stage_id = config.run_name
    else: 
        stage_id = "stage"+str(os.listdir(os.path.join(config.save_path, unique_id))-1)
        
    save_dir = os.path.join(config.save_path, unique_id, stage_id)

    with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
        samples = pickle.load(f)
    with open(os.path.join(save_dir, 'prompt.json'), 'r') as f: 
        ground = json.load(f)

    ## evaluation
    img_dir = os.path.join(save_dir, 'images')
    eval_scores = score_fn1(ground, img_dir, save_dir, config)
    samples['eval_scores'] = eval_scores

    def get_new_unit():
        return {
            'prompt_embeds': [], 
            'timesteps': [], 
            'log_probs': [], 
            'latents': [], 
            'next_latents': [], 
            'eval_scores': [],
            "add_text_embeds": [],
            "add_time_ids": [],
            'prompt_embeds1_combine':[]
        }
    data = get_new_unit()

    total_batch_size = samples['eval_scores'].shape[0]
    data_size = total_batch_size // config.sample.batch_size
    for b in range(config.sample.batch_size): 
        cur_sample_num = 1 
        batch_samples = {k:v[torch.arange(b, total_batch_size, config.sample.batch_size)] for k,v in samples.items()}
        
        t_left = config.sample.num_steps - config.split_step
        t_right = config.sample.num_steps

        prompt_embeds = batch_samples['prompt_embeds'][torch.arange(0, data_size, cur_sample_num)]
        timesteps = batch_samples['timesteps'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        log_probs = batch_samples['log_probs'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        latents = batch_samples['latents'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        next_latents = batch_samples['next_latents'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        add_text_embeds = batch_samples['add_text_embeds'][torch.arange(0, data_size, cur_sample_num)]
        add_time_ids = batch_samples['add_time_ids'][torch.arange(0, data_size, cur_sample_num)]
        prompt_embeds1_combine= batch_samples['prompt_embeds1_combine'][torch.arange(0, data_size, cur_sample_num)]

        score = batch_samples['eval_scores'][torch.arange(0, data_size, cur_sample_num)]
        score = score.reshape(-1, config.split_time)
        max_idx = score.argmax(dim=1)
        min_idx = score.argmin(dim=1)
        for j, s in enumerate(score): 
            for p_n in range(2):
                if p_n==0 and s[max_idx[j]] >= config.eval.pos_threshold: 
                    used_idx = max_idx[j]
                    used_idx_2 = j*config.split_time+max_idx[j]
                elif p_n==1 and s[min_idx[j]] < config.eval.neg_threshold: 
                    used_idx = min_idx[j]
                    used_idx_2 = j*config.split_time+min_idx[j]
                else: 
                    continue

                data['prompt_embeds'].append(prompt_embeds[used_idx_2]) # j*split_times[i-1]
                data['timesteps'].append(timesteps[used_idx_2]) # j*split_times[i-1]

                data['log_probs'].append(log_probs[used_idx_2])
                data['latents'].append(latents[used_idx_2])
                data['next_latents'].append(next_latents[used_idx_2])
                data['eval_scores'].append(s[used_idx])
                data['add_text_embeds'].append(add_text_embeds[used_idx_2])
                data['add_time_ids'].append(add_time_ids[used_idx_2])
                data['prompt_embeds1_combine'].append(prompt_embeds1_combine[used_idx_2])

        cur_sample_num *= config.split_time

    if len(data.keys()) != 0:
        data = {k:torch.stack(v, dim=0) for k,v in data.items()}

    print("train_data.shape:")
    for k,v in data.items():
        print(f"{k}:", v.shape)

    with open(os.path.join(save_dir, 'sample_stage.pkl'), 'wb') as f:
        pickle.dump(data, f)

def run_train_main():
    # basic Accelerate and logging setup
    config = FLAGS.config

    torch.cuda.set_device(config.dev_id)

    if config.exp_name:
        unique_id = config.exp_name
    else: 
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)

    if config.run_name:
        stage_id = config.run_name
    else: 
        stage_id = "stage"+str(os.listdir(os.path.join(config.save_path, unique_id)))
        
    save_dir = os.path.join(config.save_path, unique_id, stage_id)

    if config.resume_from:
        print("loading model. Please Wait.")
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print("load successfully!")

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    num_train_timesteps_2 = int(config.split_step * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir, # os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps_2,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="d3po-pytorch", config=config.to_dict(), init_kwargs={"wandb": {"name": unique_id+"_"+stage_id}}
        )
    logger.info(f"\n{config}")

    seed_everything(config.seed)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionXLPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16) ## float16
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    
    rank=4
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.unet.add_adapter(unet_lora_config)

    # set up diffusers-friendly checkpoint saving with Accelerate

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(pipeline.unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(unwrap_model(model), type(unwrap_model(pipeline.text_encoder))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(unwrap_model(model), type(unwrap_model(pipeline.text_encoder_2))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(pipeline.unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(pipeline.text_encoder))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(pipeline.text_encoder_2))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
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
        if accelerator.mixed_precision == "fp16":
            models = [unet_]
            cast_training_params(models, dtype=torch.float32)
    

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    for param in pipeline.unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids
    
        # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            if tokenizers is not None:
                tokenizer = tokenizers[i]
                text_input_ids = tokenize_prompt(tokenizer, prompt)
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    neg_prompt_embed,_= encode_prompt([pipeline.text_encoder, pipeline.text_encoder_2], [pipeline.tokenizer, pipeline.tokenizer_2], "")
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    pipeline.unet, optimizer = accelerator.prepare(pipeline.unet, optimizer)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.train.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        
    # get sample dict
    samples = load_sample_stage(save_dir)

    accelerator.save_state()

    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
    init_samples = copy.deepcopy(samples)
    LossRecord = []
    GradRecord = []
    for epoch in range(config.train.num_epochs):
        # shuffle samples along batch dimension
        samples = {}
        LossRecord.append([])
        GradRecord.append([])

        total_batch_size = init_samples["eval_scores"].shape[0]
        perm = torch.randperm(total_batch_size)
        samples = {k: v[perm] for k, v in init_samples.items()}

        perms = torch.stack( # v2["timesteps"].shape[1]
            [torch.randperm(init_samples["timesteps"].shape[1]) for _ in range(total_batch_size)]
        )
        for key in ["latents", "next_latents", "log_probs", "timesteps"]:
            samples[key] = samples[key][torch.arange(total_batch_size)[:, None], perms]
                    
        # training
        pipeline.unet.train()
        for idx in tqdm(range(0,total_batch_size//2*2,config.train.batch_size),
                    desc="Update",
                    position=2,
                    leave=False, 
                        ):

            LossRecord[epoch].append([])
            GradRecord[epoch].append([])

            sample = tree.map_structure(lambda value: value[idx:idx+config.train.batch_size].to(accelerator.device), samples)

            # cfg, classifier-free-guidance
            if config.train.cfg:
                #embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                embeds= sample["prompt_embeds1_combine"]
            else:
                embeds = sample["prompt_embeds"]
            #embeds = sample["prompt_embeds"]
            
            for t in tqdm(
                range(sample["timesteps"].shape[1]),
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):  
                
                evaluation_score = sample["eval_scores"][:]

                with accelerator.accumulate(pipeline.unet):
                    with autocast():
                        if config.train.cfg:
                            unet_added_conditions = {"time_ids": sample["add_time_ids"],"text_embeds":sample["add_text_embeds"][0]}
                            #print(sample["add_time_ids"].shape, sample["add_text_embeds"][0].shape,embeds.shape)
                            noise_pred = pipeline.unet(
                                torch.cat([sample["latents"][:, t]] * 2),
                                #sample["latents"][:, t],
                                #torch.cat([sample["timesteps"][:, t]] * 2),
                                sample["timesteps"][:, t],
                                encoder_hidden_states=embeds[0],
                                added_cond_kwargs=unet_added_conditions
                            ).sample
                            
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        else:
                            noise_pred = pipeline.unet(
                                sample["latents"][:, t], sample["timesteps"][:, t], embeds, added_cond_kwargs=unet_added_conditions
                            ).sample

                        _, total_prob, _ = ddim_step_with_logprob(
                            pipeline.scheduler,
                            noise_pred,
                            sample["timesteps"][:, t],
                            sample["latents"][:, t],
                            eta=config.sample.eta,
                            prev_sample=sample["next_latents"][:, t],
                        )
                        total_ref_prob = sample["log_probs"][:, t]

                ratio = torch.exp(total_prob - total_ref_prob)
                temp_beta1 = torch.ones_like(evaluation_score)*config.train.beta1
                temp_beta2 = torch.ones_like(evaluation_score)*config.train.beta2
                sample_weight = torch.where(evaluation_score>0, temp_beta1, temp_beta2)
                advantages = torch.clamp(
                            evaluation_score,
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )*sample_weight
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - config.train.eps,
                    1.0 + config.train.eps,
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                accelerator.backward(loss)
                total_norm = None
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(params_to_optimize, config.train.max_grad_norm)
                LossRecord[epoch][idx//config.train.batch_size].append(loss.cpu().item())
                GradRecord[epoch][idx//config.train.batch_size].append(total_norm.cpu().item() if total_norm is not None else None)
                optimizer.step()
                optimizer.zero_grad()
                
        if (epoch+1) % config.train.save_interval == 0 :
            accelerator.save_state()

    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'loss.json'),'w') as f:
        json.dump(LossRecord, f)
    with open(os.path.join(save_dir, 'eval', 'grad.json'),'w') as f:
        json.dump(GradRecord, f)
    

def main(_):
    run_sample_main()
    run_select_main()
    run_train_main()

if __name__ == "__main__":
    app.run(main)