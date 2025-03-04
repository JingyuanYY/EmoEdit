from transformers import (
    Blip2QFormerConfig, Blip2Config,
    Blip2QFormerModel, Blip2Processor,
    CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
)
import torch
import torch.multiprocessing as mp
from dataset import EmoEditDataset_MultiData
from diffusers import StableDiffusionInstructPix2PixPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
import math
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
import os
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import shutil
from config import parse_args
import utils
from model import CombinedModel
import sampler


def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# mp.set_start_method('spawn', force=True)
# torch.autograd.set_detect_anomaly(True)

def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    # writer = SummaryWriter(log_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-large-patch14")
    # tokenizer = processor.tokenizer
    # txt_tower_model = CLIPTextModel.from_pretrained("/mnt/d/model/clip-vit-large-patch14")
    vision_tower_model = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-large-patch14")
    # linear_project = vision_tower_model.text_projection
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )
    num_embedding = args.num_embedding - 1
    model = CombinedModel(num_embedding, ln=args.use_ln)

    # query_tokens = query_tokens.to(accelerator.device)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vision_tower_model.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    train_dataset = EmoEditDataset_MultiData(
        origin_data_root=args.origin_data_root,
        edited_data_root=args.edited_data_root,
        instruction_file_path=args.instruction_file_path,
        mixed_precision=args.mixed_precision,
        processor=processor,
        vision_tower_model=vision_tower_model,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.dataloader_num_workers,
        sampler=sampler.ClassAwareSampler(train_dataset)
    )

    optimizer_cls = torch.optim.AdamW

    parameters = list(model.parameters())

    optimizer = optimizer_cls(
        parameters,  # TODO need to change
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.validation_epochs is not None:
        args.validation_steps = args.validation_epochs * len(train_dataset) // accelerator.num_processes

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )
    text_encoder, optimizer, train_dataloader, lr_scheduler, model = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler, model
    )
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # model.to(accelerator.device, dtype=weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("EmoEdit")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            args.resume_from_checkpoint = None
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    rank = accelerator.process_index
    print(f"Process {rank}, DataLoader length: {len(train_dataloader)}")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                text_embeds = batch['text_embeds'].to(accelerator.device)
                img_embeds = batch['img_embeds'].to(accelerator.device)
                normal_output = model(text_embeds, img_embeds).to(weight_dtype)

                # So, first, convert images to latent space.
                latents = vae.encode(batch["edited_image"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["origin_image"].to(weight_dtype)).latent_dist.mode()

                encoder_hidden_states = normal_output
                    # print(outputs)
                # Get the text embedding for conditioning.
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                #
                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""], tokenizer).to(accelerator.device))[0].to(weight_dtype)
                    new_encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
                    # encoder_hidden_states.register_hook(print_grad)
                    # new_encoder_hidden_states.register_hook(print_grad)
                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds
                else:
                    new_encoder_hidden_states = encoder_hidden_states
                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(concatenated_noisy_latents, timesteps, new_encoder_hidden_states, return_dict=False)[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                instruction_hidden_state = batch['instruction_hidden_state']
                instruction_loss = F.mse_loss(normal_output, instruction_hidden_state, reduction="mean")

                Loss = args.instruction_rate * instruction_loss + args.diffusion_rate * loss

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(Loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # if args.use_ema:
                #     ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                # logger.info(
                                #     f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                # )
                                # logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                for tracker in accelerator.trackers:
                    tracker.writer.add_scalar("Total_Loss", Loss, global_step)
                    tracker.writer.add_scalar("Diffusion_Loss", loss, global_step)
                    tracker.writer.add_scalar("Instruction_Loss", instruction_loss, global_step)
                        # logger.info(f"Saved state to {save_path}")
                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        # save_dir = os.path.join(args.output_dir, f'validation/validation_image-{global_step}')
                        # os.makedirs(save_dir, exist_ok=True)
                        save_dir = os.path.join(args.output_dir, f"Q-Former")
                        os.makedirs(save_dir, exist_ok=True)
                        # utils.log_validation(args=args, processor=processor, vision_tower_model=vision_tower_model,
                        #                      Q_Former=model, accelerator=accelerator,
                        #                      weight_dtype=weight_dtype,
                        #                      save_dir=save_dir, generation_pipeline=IP2P)

                        torch.save(accelerator.unwrap_model(model).state_dict(),
                                   os.path.join(args.output_dir, f"Q-Former/Q-Former_{global_step}.pth"))

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, "Q-Former.pth"))
    accelerator.end_training()


if __name__ == "__main__":
    config_dic = {
        'origin_data_root': "/mnt/d/dataset",
        'edited_data_root': "/mnt/d/dataset/EmoEdit/train_v2",
        'validation_data_root': None,
        'instruction_file_path': "/mnt/d/code/EmoEdit/GPT/Reason_final.csv",
        'max_train_steps': 30000,
        'num_train_epochs': 1000,
        'seed': 47500,
        'diffusion_rate': 0.1,
        'instruction_rate': 1,
        'learning_rate': 0.0001,
        'output_dir': '/mnt/d/code/EmoEdit/CVPR/train_data/11-03_Rate10_LR-4_Newdataset_Seed-48000',
        'num_embedding': 77,
        'batch_size': 32,
        'use_ln': None,
    }
    os.makedirs(config_dic['output_dir'], exist_ok=True)
    filename_with_extension = __file__

    # 获取不包括扩展名的文件名
    filename_without_extension = os.path.basename(__file__).split('.')[0]
    save_dic = config_dic.copy()
    save_dic['file_name'] = filename_without_extension

    utils.save_text_to_file(save_dic, os.path.join(save_dic['output_dir'], 'config.txt'))
    args = parse_args(**config_dic)
    main(args)
