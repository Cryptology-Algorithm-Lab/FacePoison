"""
Unified poison generator training:
  L_total = L_emb + λ_del * L_del + λ_con * L_con + λ_attr * L_attr + λ_aug * L_aug_con

- Single surrogate (PartialFC)
- attr_repr: CPU-resident lookup, NOT loaded via DataLoader (memory-safe)
- Detailed per-term logging every cfg.frequent steps + wandb
"""

import argparse
import logging
import os
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torchvision.transforms.v2 import (
    Normalize, ColorJitter, GaussianBlur, Grayscale,
    RandomErasing, RandomChoice,
)

import mxnet as mx
import numbers

from backbones import get_model
from backbones.custom_generator import ResNetGenerator
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import (
    setup_seed, get_dist_info, DistributedSampler, worker_init_fn,
)
from utils.utils_logging import AverageMeter, init_logging


assert torch.__version__ >= "1.12.0"


# ============================================================
# Distributed init
# ============================================================
try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank, local_rank, world_size = 0, 0, 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank, world_size=world_size,
    )


# ============================================================
# Hyper-parameters
# ============================================================
STAGE1_ITERS = 200
LAMBDA_DEL   = 0.3
LAMBDA_CON   = 5.0
LAMBDA_ATTR  = 10.0
LAMBDA_AUG   = 10.0
GEN_LR       = 1e-3


# ============================================================
# Dataset: returns (img, label, dataset_index) — NO attr
# ============================================================
class IndexedMXFaceDataset(Dataset):
    """MXFaceDataset that also returns the dataset index (for attr lookup)."""

    def __init__(self, root_dir, local_rank):
        super().__init__()
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if isinstance(label, numbers.Number):
            label = int(label)
        elif isinstance(label, (list, tuple, np.ndarray)):
            label = int(label[0])
        elif isinstance(label, np.generic):
            label = int(label.item())
        else:
            raise TypeError(f"Unexpected label type: {type(label)}")
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, index          # <-- index for attr lookup

    def __len__(self):
        return len(self.imgidx)


# ============================================================
# DataLoaderX (CUDA-stream prefetch)
# ============================================================
import queue as Queue
import threading


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super().__init__(daemon=True)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(
                    device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


# ============================================================
# Attr align loss (memory-safe version)
# ============================================================
def attr_align_loss(f_x_delta, f_x, attr_repr):
    residual = (f_x_delta - f_x)

    residual = residual - residual.mean(dim=0, keepdim=True)
    attr_repr = attr_repr - attr_repr.mean(dim=0, keepdim = True)

    residual = F.normalize(residual)
    attr_repr = F.normalize(attr_repr)

    # B X B each
    B = f_x.size(0)
    dist_mat_res = torch.mm(residual, residual.T)
    dist_mat_attr = torch.mm(attr_repr, attr_repr.T)

    loss = (dist_mat_res - dist_mat_attr).abs().sum() / (B * (B-1))
    return loss 

# ============================================================
# Augmentation
# ============================================================
GS = Grayscale(num_output_channels=3)
GB = GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
CJ = ColorJitter(brightness=[0.1, 0.5], hue=[-0.1, 0.3],
                  contrast=[0.3, 0.6], saturation=[0.2, 0.5])
RE = RandomErasing(1)
aug_func = RandomChoice([GS, GB, CJ, RE], [0.25, 0.25, 0.25, 0.25])


@torch.no_grad()
def augmentation(imgs):
    imgs = imgs * 0.5 + 0.5    
    aug_imgs = aug_func(imgs)
    return (aug_imgs - 0.5) / 0.5


# ============================================================
# Main
# ============================================================
def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)

    cfg.output = os.path.join(cfg.output, args.suffix)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0 else None
    )

    wandb_logger = None
    run_name = ""
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print(f"WandB: {e}")
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = (run_name if cfg.suffix_run_name is None
                    else run_name + f"_{cfg.suffix_run_name}")
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity, project=cfg.wandb_project,
                sync_tensorboard=True, resume=cfg.wandb_resume,
                name=run_name, notes=cfg.notes,
            ) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print(f"WandB init: {e}")

    # ----------------------------------------------------------
    # Data: full or top-K class subset
    # ----------------------------------------------------------
    base_dataset = IndexedMXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    use_subset = args.subset_classes > 0

    if use_subset:
        labels_all = torch.load("")  # TODO: path to .pt of MS1MV3 per-sample labels (only used with --subset_classes)
        unique_labels, cts = labels_all.unique(return_counts=True)
        K = args.subset_classes
        cfg.num_classes = K
        cfg.num_image = int(cts[:K].sum().item())

        top_k_classes = set(range(K))
        valid_indices = [i for i, lb in enumerate(labels_all.numpy())
                         if int(lb) in top_k_classes]
        train_ds = Subset(base_dataset, valid_indices)
        valid_indices_np = np.array(valid_indices, dtype=np.int64)

        if rank == 0:
            logging.info(f"[Subset mode] {len(valid_indices)} samples from top {K} classes")
    else:
        # Full dataset — valid_indices is identity mapping
        N = len(base_dataset)
        cfg.num_image = N
        train_ds = base_dataset
        valid_indices_np = np.arange(N, dtype=np.int64)

        if rank == 0:
            logging.info(f"[Full mode] {N} samples, {cfg.num_classes} classes")

    rank_, world_size_ = get_dist_info()
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size_, rank=rank_,
        shuffle=True, seed=cfg.seed,
    )
    init_fn = partial(worker_init_fn, num_workers=cfg.num_workers,
                      rank=rank_, seed=cfg.seed)
    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    # ----------------------------------------------------------
    # Attr repr: CPU-resident, index-based lookup (NO DataLoader)
    #   - Only the batch slice is moved to GPU per step, then freed
    # ----------------------------------------------------------
    attr_repr_cpu = torch.load(cfg.attr_repr_dir, map_location="cpu")
    if rank == 0:
        logging.info(
            f"attr_repr loaded on CPU: shape={attr_repr_cpu.shape}, "
            f"dtype={attr_repr_cpu.dtype}, "
            f"mem={attr_repr_cpu.element_size() * attr_repr_cpu.nelement() / 1e9:.2f} GB"
        )

    # ----------------------------------------------------------
    # Backbone + PartialFC head
    # ----------------------------------------------------------
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16,
        num_features=cfg.embedding_size,
    ).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False,
        device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True,
    )
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.train()
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2],
        cfg.interclass_filtering_threshold,
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Schedule
    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step  = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step,
    )

    start_epoch = 0
    global_step = 0

    if cfg.resume:
        ckpt = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        backbone.module.load_state_dict(ckpt["state_dict_backbone"])
        module_partial_fc.load_state_dict(ckpt["state_dict_softmax_fc"])
        opt.load_state_dict(ckpt["state_optimizer"])
        lr_scheduler.load_state_dict(ckpt["state_lr_scheduler"])
        del ckpt

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # Log unified hyper-parameters
    if rank == 0:
        logging.info("=" * 50)
        logging.info("[Unified] STAGE1_ITERS = %d", STAGE1_ITERS)
        logging.info("[Unified] LAMBDA_DEL   = %.2f", LAMBDA_DEL)
        logging.info("[Unified] LAMBDA_CON   = %.2f", LAMBDA_CON)
        logging.info("[Unified] LAMBDA_ATTR  = %.2f", LAMBDA_ATTR)
        logging.info("[Unified] LAMBDA_AUG   = %.2f", LAMBDA_AUG)
        logging.info("[Unified] GEN_LR       = %.1e", GEN_LR)
        logging.info("[Unified] DATA_MODE    = %s",
                     f"subset(top-{args.subset_classes})" if use_subset else "full")
        logging.info("=" * 50)

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=wandb_logger,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent, total_step=cfg.total_step,
        batch_size=cfg.batch_size, start_step=global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()
    amp_scaler = GradScaler(enabled=cfg.fp16, growth_interval=100)
    amp_G = GradScaler(enabled=cfg.fp16, growth_interval=100)

    # ----------------------------------------------------------
    # Generator (poison module)
    # ----------------------------------------------------------
    poison_module = ResNetGenerator().cuda()
    poison_module = torch.nn.parallel.DistributedDataParallel(
        module=poison_module, broadcast_buffers=False,
        device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True,
    )
    poison_module.register_comm_hook(None, fp16_compress_hook)
    opt_P = torch.optim.Adam(params=poison_module.parameters(), lr=GEN_LR)
    
    
    

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(start_epoch, cfg.num_epoch):

        if rank == 0:
            logging.info(f"{'='*20} Epoch {epoch} start {'='*20}")

        train_loader.sampler.set_epoch(2 * epoch)
        it = iter(train_loader)

        # ========================================================
        # Stage 1: Surrogate training (generator frozen)
        # ========================================================
        backbone.train()
        module_partial_fc.train()
        poison_module.eval()

        if rank == 0:
            logging.info(f"[E{epoch}] Stage 1 start ({STAGE1_ITERS} iters)")
        t_s1 = time.time()
        
        # BUG FIX: DO NOT APPLY NOISE ON THE FIRST EPOCH
        APPLY_POISON = (epoch > 0)

        for s1_step in range(STAGE1_ITERS):
            try:
                img, local_labels, _ = next(it)
            except StopIteration:
                it = iter(train_loader)
                img, local_labels, _ = next(it)
            
            # BUG FIX: DO NOT APPLY NOISE ON THE FIRST EPOCH
            if APPLY_POISON:
                with torch.no_grad():
                    noise = poison_module(img)
                    poisoned_img = (img + noise).clamp(-1, 1)
            else:
                poisoned_img = img
            
            local_emb = backbone(poisoned_img)
            local_emb = local_emb.float()  # fp32 for PartialFC AllGather
            loss_s1: torch.Tensor = module_partial_fc(local_emb, local_labels)

            if cfg.fp16:
                opt.zero_grad()
                amp_scaler.scale(loss_s1).backward()
                amp_scaler.unscale_(opt)
            else:
                opt.zero_grad()
                loss_s1.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            if cfg.fp16:
                amp_scaler.step(opt)
                amp_scaler.update()
            else:
                opt.step()

            if rank == 0 and s1_step % 10 == 0:
                logging.info(
                    f"[E{epoch}] S1 {s1_step}/{STAGE1_ITERS}  "
                    f"loss={loss_s1.item():.4f}  "
                    f"elapsed={time.time() - t_s1:.1f}s"
                )
            if wandb_logger:
                wandb_logger.log({
                    'S1/loss': loss_s1.item(),
                    'Process/Epoch': epoch,
                })

        if rank == 0:
            logging.info(f"[E{epoch}] Stage 1 done ({time.time() - t_s1:.1f}s)  "
                         f"last_loss={loss_s1.item():.4f}")

        del it
        torch.cuda.empty_cache()

        # ========================================================
        # Stage 2: Generator training (backbone frozen)
        # ========================================================
        backbone.eval()
        module_partial_fc.eval()
        poison_module.train()

        train_loader.sampler.set_epoch(2 * epoch + 1)

        stage2_steps = len(train_loader)
        if rank == 0:
            logging.info(f"[E{epoch}] Stage 2 start ({stage2_steps} iters)")
        t_s2 = time.time()

        for s2_step, (img, local_labels, ds_indices) in enumerate(train_loader):
            global_step += 1

            # ---- Attr: CPU → GPU (batch only) ----
            # ds_indices are indices into the Subset; map back to base dataset
            base_indices = valid_indices_np[ds_indices.cpu().numpy()]
            attr_batch = attr_repr_cpu[base_indices].cuda(non_blocking=True)

            # ---- Forward ----
            noise = poison_module(img)
            poisoned_img = (img + noise).clamp(-1, 1)
            
            aug_img = augmentation(img)
            aug_noise = poison_module(aug_img)            
            poisoned_aug = (aug_img + aug_noise).clamp(-1, 1)

            # Backbone forward: poisoned, delta, clean (batched)
            # Better optimization: Make batch size the power of two.
            combined = torch.cat([poisoned_img, noise, img, poisoned_aug], dim=0)
            emb_all = backbone(combined).float()  # fp32 for PartialFC AllGather
            B = img.size(0)
            emb_poi     = emb_all[:B]       # f(x + δ)
            emb_delta   = emb_all[B:2*B]    # f(δ)
            emb_clean   = emb_all[2*B:3*B]  # f(x)
            emb_aug_poi = emb_all[3*B:]     # f(x' + δ')

            # --- L_emb: poisoned classification ---
            loss_emb = module_partial_fc(emb_poi, local_labels)

            # --- L_del: δ-only classification ---
            loss_del = module_partial_fc(emb_delta, local_labels)

            # --- L_con: cosine(f(x+δ), f(δ)) → 1 ---
            loss_con = 1 - F.cosine_similarity(emb_poi, emb_delta, dim=-1).mean()

            # --- L_attr: attr structure alignment ---
            loss_attr = attr_align_loss(emb_poi, emb_clean, attr_batch)

            # --- L_aug_con: augmentation invariance ---
#             aug_img = augmentation(img)
#             aug_noise = poison_module(aug_img)
#             poisoned_aug = (aug_img + aug_noise).clamp(-1, 1)
#             emb_aug_poi = backbone(poisoned_aug).float()

            loss_aug_con = 1 - F.cosine_similarity(
                F.normalize(emb_poi), F.normalize(emb_aug_poi), dim=-1
            ).mean()

            # --- Total ---
            loss_total = (
                loss_emb
                + LAMBDA_DEL  * loss_del
                + LAMBDA_CON  * loss_con
                + LAMBDA_ATTR * loss_attr
                + LAMBDA_AUG  * loss_aug_con
            )

            # ---- Backward ----
            if cfg.fp16:
                opt_P.zero_grad()
                amp_G.scale(loss_total).backward()
                amp_G.unscale_(opt_P)
                torch.nn.utils.clip_grad_norm_(poison_module.parameters(), 5)
                amp_G.step(opt_P)
                amp_G.update()
            else:
                opt_P.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(poison_module.parameters(), 5)
                opt_P.step()

            # ---- Free attr batch immediately ----
            del attr_batch

            # ---- Logging ----
            with torch.no_grad():
                loss_emb_v     = loss_emb.item()
                loss_del_v     = loss_del.item()
                loss_con_v     = loss_con.item()
                loss_attr_v    = loss_attr.item()
                loss_aug_con_v = loss_aug_con.item()
                loss_total_v   = loss_total.item()
                noise_linf     = noise.abs().max().item()
                noise_abs_mean = noise.abs().mean().item()

                if wandb_logger:
                    wandb_logger.log({
                        'S2/total':       loss_total_v,
                        'S2/L_emb':       loss_emb_v,
                        'S2/L_del':       loss_del_v,
                        'S2/L_con':       loss_con_v,
                        'S2/L_attr':      loss_attr_v,
                        'S2/L_aug_con':   loss_aug_con_v,
                        'Noise/linf':     noise_linf,
                        'Noise/abs_mean': noise_abs_mean,
                        'AMP/scale':      amp_G.get_scale(),
                        'Process/Step':   global_step,
                        'Process/Epoch':  epoch,
                    })

                loss_am.update(loss_total_v, 1)
                callback_logging(
                    global_step, loss_am, epoch, cfg.fp16,
                    lr_scheduler.get_last_lr()[0], amp_G,
                )

                # Detailed per-term stdout logging (every 100 steps)
                if rank == 0 and global_step % 100 == 0:
                    logging.info(
                        f"[E{epoch}] S2 step {global_step}  "
                        f"total={loss_total_v:.4f}  amp_scale={amp_G.get_scale():.0f}\n"
                        f"    L_emb     = {loss_emb_v:.4f}\n"
                        f"    L_del     = {loss_del_v:.4f}  (x{LAMBDA_DEL} = {LAMBDA_DEL * loss_del_v:.4f})\n"
                        f"    L_con     = {loss_con_v:.4f}  (x{LAMBDA_CON} = {LAMBDA_CON * loss_con_v:.4f})\n"
                        f"    L_attr    = {loss_attr_v:.4f}  (x{LAMBDA_ATTR} = {LAMBDA_ATTR * loss_attr_v:.4f})\n"
                        f"    L_aug_con = {loss_aug_con_v:.4f}  (x{LAMBDA_AUG} = {LAMBDA_AUG * loss_aug_con_v:.4f})\n"
                        f"    noise: linf={noise_linf:.4f}  abs_mean={noise_abs_mean:.4f}"
                    )

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)
                    backbone.eval()

        t_s2_total = time.time() - t_s2
        if rank == 0:
            logging.info(f"[E{epoch}] Stage 2 done ({t_s2_total:.1f}s)")

        # ---- Save ----
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(poison_module.module.state_dict(), path_module)
            if wandb_logger and cfg.save_artifacts:
                import wandb
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if hasattr(cfg, 'dali') and cfg.dali:
            train_loader.reset()

    # Final save
    if rank == 0:
        path_module = os.path.join(cfg.output, "model_final.pt")
        torch.save(poison_module.module.state_dict(), path_module)
        if wandb_logger and cfg.save_artifacts:
            import wandb
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Unified poison generator training")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("suffix", type=str, help="output suffix")
    parser.add_argument("--subset_classes", type=int, default=0,
                        help="Number of top-K classes to use. 0 = full dataset.")
    main(parser.parse_args())
