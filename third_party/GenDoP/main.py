import os
import tyro
import math
import time
import shutil
from functools import partial

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

from core.options import AllConfigs
from core.models import LMM
from core.provider import ShotTrajDataset, collate_fn
from core.utils import init_logger

import kiui

# torch.autograd.set_detect_anomaly(True)

def main():    
    opt = tyro.cli(AllConfigs)
    print("save_epoch:", opt.save_epoch)
    print("pose_length:", opt.pose_length)
    # validate options
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    os.makedirs(os.path.join(opt.workspace, opt.exp_name), exist_ok=True)
    logfile = os.path.join(opt.workspace, opt.exp_name, 'log.txt')
    logger = init_logger(logfile)

    # print options
    accelerator.print(opt)
    
    # tokenizer
    vocab_size = opt.discrete_bins + 4 # discrete_bins+1, bos, eos, pad

    # model
    model = LMM(opt)

    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    # specially handle positional embeddings: if we finetune from uncond models, the weight can be aligned to the right, otherwise to the left.
                    if 'mesh_decoder.model.embed_positions.weight' in k and v.shape[1] == state_dict[k].shape[1]:
                        if state_dict[k].shape[0] > v.shape[0]:
                            if opt.align_posemb == 'right':
                                state_dict[k][-v.shape[0]:] = v
                            else:
                                state_dict[k][:v.shape[0]] = v
                            logger.warning(f'embed_positions: aligning positional embeddings {v.shape} --> {state_dict[k].shape}.')
                        else:
                            if opt.align_posemb == 'left':
                                state_dict[k] = v[:state_dict[k].shape[0]]
                            else:
                                state_dict[k] = v[-state_dict[k].shape[0]:]
                            logger.warning(f'embed_positions: aligning positional embeddings {v.shape} --> {state_dict[k].shape}.')
                    else:
                        logger.warning(f'mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                logger.warning(f'unexpected param {k}: {v.shape}')
    
    # count params
    num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f'trainable param num: {num_p/1024/1024:.6f} M, total param num: {total_p/1024/1024:.6f}')

    train_dataset = ShotTrajDataset(opt, training=True)
    
    logger.info(f'train dataset size: {len(train_dataset)}')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, opt=opt),
    )

    test_dataset = ShotTrajDataset(opt, training=False)

    logger.info(f'test dataset size: {len(test_dataset)}')
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, opt=opt),
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01, betas=(0.9, 0.95))

    total_steps = opt.num_epochs * len(train_dataloader) // opt.gradient_accumulation_steps
    def _lr_lambda(current_step, warmup_ratio=opt.warmup_ratio, num_cycles=0.5, min_ratio=0.1):
        progress = current_step / max(1, total_steps)
        if warmup_ratio > 0 and progress < warmup_ratio:
            return progress / warmup_ratio
        progress = (progress - warmup_ratio) / (1 - warmup_ratio)
        return max(min_ratio, min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # wandb
    if opt.use_wandb and accelerator.is_main_process:
        import wandb # set WAND_API_KEY in env
        wandb.init(project='lmm', name=opt.workspace.replace('workspace_', ''), config=opt)

    # loop
    old_save_dirs = []
    best_loss = 1e9
    for epoch in range(opt.start_epoch, opt.num_epochs):

        save_dir = os.path.join(opt.workspace, opt.exp_name, f'ep{epoch:04d}')

        # train
        if not opt.debug_eval:
            model.train()
            total_loss = 0
            t_start = time.time()
            for i, data in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()

                    step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs
                    step_ratio = opt.resume_step_ratio + (1 - opt.resume_step_ratio) * step_ratio

                    out = model(data, step_ratio)
                    loss = out['loss']

                    accelerator.backward(loss)

                    # gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                    optimizer.step()
                    scheduler.step()

                    total_loss += out['loss'].detach()

                if accelerator.is_main_process:
                    # logging
                    if i % 10 == 0:
                        mem_free, mem_total = torch.cuda.mem_get_info()
                        log = f"{epoch:03d}:{i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} loss: {loss.item():.6f}"
                        if 'loss_ce' in out:
                            log += f" loss_ce: {out['loss_ce'].item():.6f}"
                        if 'loss_kl' in out:
                            log += f" loss_kl: {out['loss_kl'].item():.6f}"
                        logger.info(log)

            total_loss = accelerator.gather_for_metrics(total_loss).mean().item()
            torch.cuda.synchronize()
            t_end = time.time()
            if accelerator.is_main_process:
                total_loss /= len(train_dataloader)
                logger.info(f"Train epoch: {epoch} loss: {total_loss:.6f} time: {(t_end - t_start)/60:.2f}min")
            
                # wandb
                if opt.use_wandb:
                    wandb.log({'train_loss': total_loss})
            
            if total_loss < best_loss:
                best_loss = total_loss
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.save_model(model, os.path.join(opt.workspace, opt.exp_name))
                    shutil.copy(os.path.join(os.path.join(opt.workspace, opt.exp_name), 'model.safetensors'), os.path.join(opt.workspace, opt.exp_name, 'best.safetensors'))

            if epoch % opt.save_epoch == 0 or epoch == opt.num_epochs - 1:
                os.makedirs(save_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                accelerator.save_model(model, save_dir)
        else:
            if accelerator.is_main_process:
                logger.info(f"epoch: {epoch} skip training for debug !!!")

        # eval
        if opt.eval_mode == 'loss':
            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, data in enumerate(test_dataloader):
                    out = model(data)
                    loss = out['loss']

                    if accelerator.process_index < 4 and i < 4:
                        if opt.cond_mode == 'image':
                            image = data['rgb'][0].detach().cpu().numpy().transpose(1, 2, 0)
                            kiui.write_image(f'{save_dir}/test_ep{epoch}_proc{accelerator.process_index}_{i}_img.png', image)
                        masks = data['masks'][0].detach().cpu().numpy()
                        coords = data['labels'][0].detach().cpu().numpy()[masks][1+opt.num_cond_tokens:-1]
                        pred_coords = out['logits'][0].argmax(-1).detach().cpu().numpy()[masks][opt.num_cond_tokens:-2]
                        
                    total_loss += loss.detach()

                total_loss = accelerator.gather_for_metrics(total_loss).mean()
                if accelerator.is_main_process:
                    total_loss /= len(test_dataloader)
                    logger.info(f"Eval epoch: {epoch} loss: {total_loss:.6f}")
        
        elif opt.eval_mode == 'generate':
            model.eval()
            unwrapped_model = accelerator.unwrap_model(model)
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    for i, data in enumerate(test_dataloader):
                        conds = data['conds'] # [B, 3, H, W] or [B, N, 6]
                        meshes, tokens = unwrapped_model.generate(conds, num_faces=opt.test_num_face, tokenizer=tokenizer)

                        # if accelerator.process_index < 4:
                        if opt.cond_mode == 'image':
                            image = data['conds'][0].detach().cpu().numpy().transpose(1, 2, 0)
                            kiui.write_image(f'{save_dir}/testgen_ep{epoch}_proc{accelerator.process_index}_{i}_img.png', image)
                        masks = data['masks'][0].detach().cpu().numpy()
                        coords = data['labels'][0].detach().cpu().numpy()[masks][1+opt.num_cond_tokens:-1]

                if accelerator.is_main_process:
                    logger.info(f"Eval epoch: {epoch} generated meshes saved.")
        else:
            if accelerator.is_main_process:
                logger.info(f"Eval epoch: {epoch} skip evaluation.")
            

if __name__ == "__main__":
    main()
