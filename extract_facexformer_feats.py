"""
Extract and save FaceXFormer encoder features for MS1MV3.

- 4-GPU DataParallel, no grad, fp16 inference
- shuffle=False / SequentialSampler so the output order matches the dataset
- FaceXFormer preprocessing: 224x224, ImageNet normalization
  (NOT face-recognition normalization)
- Output: (N, 768) float32 tensor, N = dataset size

Usage:
    python extract_facexformer_feats.py \
        --rec_dir /path/to/ms1m_v3 \
        --facex_repo ./facexformer \
        --facex_ckpt ckpts/model.pt \
        --output ms1mv3_facexformer_feats.pt \
        --batch_size 256
"""

import argparse
import numbers
import os
import sys
import time

import mxnet as mx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# Dataset: MXFace for FaceXFormer (224x224, ImageNet norm)
# ============================================================
class MXFaceForFaceX(Dataset):
    """
    MS1MV3 RecordIO → FaceXFormer-compatible preprocessing.

    CRITICAL transform differences vs. face-recognition training:
      Face-Recog: 112x112, Normalize(0.5, 0.5)   → range [-1, 1]
      FaceXFormer: 224x224, ImageNet norm          → range ~[-2.1, 2.6]

    Inference-only; no RandomHorizontalFlip.
    """

    def __init__(self, root_dir):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
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

        sample = mx.image.imdecode(img).asnumpy()   # (H, W, 3) uint8 RGB
        sample = self.transform(sample)
        return sample, label, index

    def __len__(self):
        return len(self.imgidx)


# ============================================================
# Transform verification
# ============================================================
def verify_transforms(dataset, facex_model, device):
    """Sanity-check the transform pipeline on the first sample:
    output shape (3, 224, 224), values in ImageNet-norm range, and a
    successful forward pass through FaceXFormer.
    """
    print("=" * 50)
    print("[Transform Verification]")

    img, label, idx = dataset[0]
    assert img.shape == (3, 224, 224), f"Shape mismatch: {img.shape}, expected (3, 224, 224)"
    print(f"  Shape: {img.shape} OK")

    vmin, vmax = img.min().item(), img.max().item()
    assert vmin < -1.0, f"Min value {vmin:.3f} too high — ImageNet norm should produce values < -1"
    assert vmax > 1.0, f"Max value {vmax:.3f} too low — ImageNet norm should produce values > 1"
    print(f"  Value range: [{vmin:.3f}, {vmax:.3f}] OK (ImageNet-normalized)")

    # Channel-wise mean check (should be ~0 after ImageNet norm)
    for ch, ch_name in enumerate(["R", "G", "B"]):
        ch_mean = img[ch].mean().item()
        print(f"  {ch_name} channel mean: {ch_mean:.3f}")

    # Forward pass test
    batch = img.unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        try:
            out = facex_model(batch)
            print(f"  Forward pass: OK")
            if isinstance(out, (tuple, list)):
                print(f"  Output type: tuple/list of {len(out)} elements")
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        print(f"    [{i}] shape={o.shape}, dtype={o.dtype}")
                    elif isinstance(o, dict):
                        print(f"    [{i}] dict with keys: {list(o.keys())}")
            elif isinstance(out, torch.Tensor):
                print(f"  Output shape: {out.shape}")
        except Exception as e:
            print(f"  Forward FAILED: {e}")
            raise

    print("  Label sample: idx={}, label={}".format(idx, label))
    print("[Verification PASSED]")
    print("=" * 50)


# ============================================================
# Feature extraction
# ============================================================
@torch.no_grad()
def extract_features(model, dataloader, total_samples, feat_dim, device):
    """Extract features for every sample, indexed by dataset position."""
    all_feats = torch.zeros(total_samples, feat_dim, dtype=torch.float32)
    all_labels = torch.zeros(total_samples, dtype=torch.long)
    filled = torch.zeros(total_samples, dtype=torch.bool)

    model.eval()
    t0 = time.time()

    for batch_idx, (imgs, labels, indices) in enumerate(tqdm(dataloader, desc="Extracting")):
        imgs = imgs.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(imgs)

        # FaceXFormer may return a dict, tuple, or a bare tensor.
        if isinstance(output, dict):
            feat = None
            for key in ['fused', 'encoder', 'backbone']:
                if key in output:
                    feat = output[key]
                    break
            if feat is None:
                for v in output.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 2:
                        feat = v
                        break
        elif isinstance(output, (tuple, list)):
            feat = output[0]
        else:
            feat = output

        if feat.dim() == 3:
            feat = feat.mean(dim=1)   # [B, N, D] → [B, D]
        elif feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])  # [B, C, H, W] → [B, C]

        feat = feat.float().cpu()
        indices = indices.long()

        all_feats[indices] = feat
        all_labels[indices] = labels
        filled[indices] = True

        if batch_idx % 200 == 0 and batch_idx > 0:
            elapsed = time.time() - t0
            speed = (batch_idx * dataloader.batch_size) / elapsed
            print(f"  [{batch_idx}/{len(dataloader)}] "
                  f"{speed:.0f} img/s | "
                  f"feat shape={feat.shape} | "
                  f"filled={filled.sum().item()}/{total_samples}")

    n_filled = filled.sum().item()
    if n_filled != total_samples:
        print(f"WARNING: {total_samples - n_filled} samples were NOT filled!")
        print(f"  Filled: {n_filled}/{total_samples}")
    else:
        print(f"All {total_samples} samples filled successfully.")

    total_time = time.time() - t0
    print(f"Total extraction time: {total_time:.1f}s ({total_samples / total_time:.0f} img/s)")

    return all_feats, all_labels


# ============================================================
# Label order verification
# ============================================================
def verify_label_order(extracted_labels, ref_labels_path=None):
    """Check that extracted labels agree with a reference label file."""
    if ref_labels_path and os.path.exists(ref_labels_path):
        ref = torch.load(ref_labels_path, map_location="cpu")
        N = min(len(ref), len(extracted_labels))
        match = (ref[:N] == extracted_labels[:N]).sum().item()
        print(f"[Label Order Check] {match}/{N} labels match "
              f"with {ref_labels_path} ({match/N*100:.2f}%)")
        if match != N:
            mismatch_idx = (ref[:N] != extracted_labels[:N]).nonzero(as_tuple=True)[0][:10]
            for i in mismatch_idx:
                print(f"  idx={i.item()}: ref={ref[i].item()} vs extracted={extracted_labels[i].item()}")
    else:
        print(f"[Label Order Check] No reference file. Saving labels for future verification.")

    unique, counts = extracted_labels.unique(return_counts=True)
    print(f"  Total samples: {len(extracted_labels)}")
    print(f"  Unique classes: {len(unique)}")
    print(f"  Samples per class: min={counts.min().item()}, "
          f"max={counts.max().item()}, avg={counts.float().mean().item():.1f}")


# ============================================================
# Main
# ============================================================
def main(args):
    print(f"PyTorch {torch.__version__} | CUDA GPUs: {torch.cuda.device_count()}")

    sys.path.insert(0, args.facex_repo)
    from network.model import FaceXFormer
    from huggingface_hub import hf_hub_download

    device = torch.device("cuda:0")

    # --- Load model ---
    if args.facex_ckpt and os.path.exists(args.facex_ckpt):
        ckpt_path = args.facex_ckpt
    else:
        print("Downloading FaceXFormer checkpoint from HuggingFace...")
        ckpt_path = hf_hub_download(
            repo_id="kartiknarayan/facexformer",
            filename="ckpts/model.pt",
            local_dir="./",
        )

    print(f"Loading FaceXFormer from {ckpt_path}")
    model = FaceXFormer()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # --- Inspect model structure (useful on first run) ---
    if args.inspect:
        print("\n[Model Structure]")
        for name, child in model.named_children():
            n_params = sum(p.numel() for p in child.parameters())
            print(f"  {name}: {type(child).__name__} ({n_params/1e6:.1f}M)")
        print()

    # --- DataParallel (4 GPU) ---
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # --- Dataset ---
    print(f"Loading MS1MV3 from {args.rec_dir}")
    dataset = MXFaceForFaceX(root_dir=args.rec_dir)
    print(f"Dataset size: {len(dataset)}")

    # --- Verify transforms ---
    raw_model = model.module if hasattr(model, 'module') else model
    verify_transforms(dataset, raw_model, device)

    # --- DataLoader (shuffle=False to preserve sample order) ---
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # --- Extract features ---
    feats, labels = extract_features(
        model=model,
        dataloader=dataloader,
        total_samples=len(dataset),
        feat_dim=args.feat_dim,
        device=device,
    )

    print(f"\nFeatures shape: {feats.shape}")
    print(f"Labels shape: {labels.shape}")

    # --- Verify label order ---
    ref_path = os.path.join(os.path.dirname(args.output), "ms1mv3_labels.pt")
    verify_label_order(labels, ref_path)

    # --- Save ---
    print(f"Saving features to {args.output}")
    torch.save(feats, args.output)

    # Save labels alongside features if no reference file existed.
    labels_path = os.path.join(os.path.dirname(args.output) or ".", "ms1mv3_labels.pt")
    if not os.path.exists(labels_path):
        print(f"Saving labels to {labels_path}")
        torch.save(labels, labels_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FaceXFormer features from MS1MV3")
    parser.add_argument("--rec_dir", type=str, required=True,
                        help="MS1MV3 RecordIO directory (contains train.rec, train.idx)")
    parser.add_argument("--facex_repo", type=str, default="./facexformer",
                        help="Path to cloned facexformer repository")
    parser.add_argument("--facex_ckpt", type=str, default="ckpts/model.pt",
                        help="FaceXFormer checkpoint path")
    parser.add_argument("--output", type=str, default="ms1mv3_facexformer_feats.pt",
                        help="Output path for extracted features")
    parser.add_argument("--feat_dim", type=int, default=768,
                        help="Feature dimension (default: 768)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size per forward (total across GPUs)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers")
    parser.add_argument("--inspect", action="store_true",
                        help="Print model structure before extraction")
    main(parser.parse_args())
