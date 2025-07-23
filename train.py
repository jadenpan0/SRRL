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
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, DDIMInverseScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
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
    
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    ####################################
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=inference_dtype) # float16
    inv_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    # pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model, torch_dtype=torch.float16)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
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
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    ###############################################################
    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        # print(models)
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers = accelerator.prepare(trainable_layers)

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
        # encode prompts
        prompt_ids1 = pipeline.tokenizer(
            prompts1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]

        # combine prompt and neg_prompt
        prompt_embeds1_combine = pipeline._encode_prompt(
            None,
            accelerator.device,
            1,
            config.sample.cfg,
            None,
            prompt_embeds=prompt_embeds1,
            negative_prompt_embeds=sample_neg_prompt_embeds
        )

        noise_latents1 = pipeline.prepare_latents(
            config.sample.batch_size, 
            pipeline.unet.config.in_channels, ## channels
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
            prompt_embeds1.dtype, 
            accelerator.device, 
            None ## generator
        )

        pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
        inv_scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)

        #@torch.no_grad()
        def prepare_resample_noise_latent(noise_latents1, resample_num, forward_guidance_scale=3.0, backward_guidance_scale=0.1):
            latents=noise_latents1
            forward_timesteps=pipeline.scheduler.timesteps
            backward_timesteps=torch.flip(forward_timesteps, [0]).to(accelerator.device)
            extra_step_kwargs={"eta": 0.0, "generator": None}
            #backward_timesteps=forward_timesteps[::-1]
            for _ in range(resample_num):
                for i,t in enumerate(forward_timesteps):
                    latents_input = torch.cat([latents] * 2) if forward_guidance_scale else latents
                    latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                    noise_pred = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
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
                            return_dict=False,
                        )[0]
                    if backward_guidance_scale:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + backward_guidance_scale * (noise_pred_text - noise_pred_uncond)
                    inv_latents = inv_scheduler.step(noise_pred, backward_timesteps[i], latents, return_dict=False)[0]
                    latents = inv_latents
            return latents
        
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

                        noise_pred = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
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
                    "images":images.cpu().detach()
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
                pil = Image.fromarray((image.to(torch.float32).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
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
                        "next_latents": new_samples["next_latents"]
                        }, f)
                else: 
                    pickle.dump({
                        "prompt_embeds": torch.cat([total_samples["prompt_embeds"], new_samples["prompt_embeds"]]), 
                        "timesteps": torch.cat([total_samples["timesteps"], new_samples["timesteps"]]), 
                        "log_probs": torch.cat([total_samples["log_probs"], new_samples["log_probs"]]), 
                        "latents": torch.cat([total_samples["latents"], new_samples["latents"]]), 
                        "next_latents": torch.cat([total_samples["next_latents"], new_samples["next_latents"]])
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
            'eval_scores': []
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
            project_name="srrl", config=config.to_dict(), init_kwargs={"wandb": {"name": unique_id+"_"+stage_id}}
        )
    logger.info(f"\n{config}")

    seed_everything(config.seed)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16) ## float16
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
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
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    
        
    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model
    

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

    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

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
                embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]
            
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
                            noise_pred = pipeline.unet(
                                torch.cat([sample["latents"][:, t]] * 2),
                                torch.cat([sample["timesteps"][:, t]] * 2),
                                embeds,
                            ).sample
                            
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        else:
                            noise_pred = pipeline.unet(
                                sample["latents"][:, t], sample["timesteps"][:, t], embeds
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
                    total_norm = accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm)
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