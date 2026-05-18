import argparse
import time
from copy import deepcopy
from PIL import Image
import numpy as np
import math
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os

import torchattacks

def _dist_is_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def _get_rank():
    return dist.get_rank() if _dist_is_avail_and_initialized() else 0

def _get_world_size():
    return dist.get_world_size() if _dist_is_avail_and_initialized() else 1

def _is_main_process():
    return _get_rank() == 0

def _setup_distributed(args):
    """
    Enable multi-GPU evaluation via torchrun (env://). This project performs per-sample
    test-time tuning, so we DO NOT wrap the model with DDP; each process runs independently
    on its shard of the dataset and we only reduce metrics at the end.
    """
    # torchrun exports: RANK, WORLD_SIZE, LOCAL_RANK
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    args.distributed = world_size > 1
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank

    if not args.distributed:
        if torch.cuda.is_available() and _is_main_process():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible is None or visible.strip() == "":
                visible = ",".join(str(i) for i in range(torch.cuda.device_count()))
            print(f"=> Visible GPUs (CUDA_VISIBLE_DEVICES): {visible}")
            print(f"=> torch.cuda.device_count(): {torch.cuda.device_count()}")
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.gpu = local_rank
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")

    # Print actual GPU binding per process (helps verify 8-GPU launch).
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible is None or visible.strip() == "":
            visible = ",".join(str(i) for i in range(torch.cuda.device_count()))
        print(
            f"=> [dist] rank={rank}/{world_size} local_rank={local_rank} "
            f"uses cuda:{local_rank} (CUDA_VISIBLE_DEVICES={visible})",
            flush=True,
        )

class _DistributedEvalSampler(torch.utils.data.Sampler):
    """
    Deterministic evaluation sampler without padding/duplication.
    Indices: rank, rank+world_size, rank+2*world_size, ...
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = int(num_replicas if num_replicas is not None else _get_world_size())
        self.rank = int(rank if rank is not None else _get_rank())

    def __iter__(self):
        n = len(self.dataset)
        return iter(range(self.rank, n, self.num_replicas))

    def __len__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - self.rank + self.num_replicas - 1) // self.num_replicas

def get_top_sim(sim_matrix):
    k = 20 # use 20 neighbor
    # guard for small number of views (e.g., when selecting a subset)
    k = int(min(k, sim_matrix.size(-1) - 1))
    if k <= 0:
        return torch.zeros(sim_matrix.size()[:-1], device=sim_matrix.device, dtype=sim_matrix.dtype)
    sim_matrix[sim_matrix>=1.0] = float('-inf')
    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    top_k_mean = top_k_values.mean(dim=-1)
    return top_k_mean

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()

def generate_augmix_views_baseline(pil_img, data_transform):
    """Produce AugMix views exactly as the original pipeline (no change to AugMix internals)."""
    return data_transform(pil_img)

def get_cls_to_patch_attention_map(model, view, out_h, out_w, device):
    """
    Return normalized CLS-to-patch attention upsampled to (out_h, out_w).
    view: (1,3,H,W) tensor on device.
    """
    with torch.no_grad():
        attn_info = model.get_attention_map(view)
    if attn_info is None:
        return None
    attn = attn_info["avg"]  # (B, g, g)
    attn = attn.unsqueeze(1)  # (B,1,g,g)
    attn = attn - attn.min()
    attn = attn / (attn.max() + 1e-8)
    attn_up = F.interpolate(attn, size=(out_h, out_w), mode='bilinear', align_corners=False)  # (B,1,H,W)
    return attn_up

def get_gradrollout_last2avg_attn_up(model, base, device, clip_quantile: float = 0.99, eps: float = 1e-8):
    """
    Compute Grad-Rollout-last2avg attention upsampled to (H,W) for attn_augmix masking.

    This is a "no-param-grad" implementation:
    - uses torch.autograd.grad to get gradients w.r.t. cam_tokens only
    - does NOT call zero_grad() or backward(), so it won't write/clear parameter .grad

    base: (1,3,H,W) tensor on device
    returns: attn_up (1,1,H,W) in [0,1], or None if unavailable
    """
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            normalized = model.normalize(base.to(dtype=model.dtype))
            # If the visual encoder is frozen (common in TTA/prompt tuning setups),
            # intermediate activations may not require grad unless the input does.
            if not normalized.requires_grad:
                normalized = normalized.detach().requires_grad_(True)
            outputs = model.image_encoder.forward_with_attn(normalized)
            if not isinstance(outputs, tuple) or len(outputs) != 4:
                return None

            features, attn_maps_all, cam_tokens_all, grid = outputs
            if not isinstance(attn_maps_all, (list, tuple)) or len(attn_maps_all) < 2:
                return None
            if not isinstance(cam_tokens_all, (list, tuple)) or len(cam_tokens_all) < 2:
                return None

            features = features / features.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                text_features = model.get_text_features()
                logit_scale = model.logit_scale.exp()
            logits = logit_scale * features @ text_features.t()  # (B,C)

            top1 = logits.detach().argmax(dim=-1)
            s = logits[torch.arange(logits.size(0), device=logits.device), top1].sum()

            attn_prev, attn_last = attn_maps_all[-2], attn_maps_all[-1]
            tokens_prev, tokens_last = cam_tokens_all[-2], cam_tokens_all[-1]
            if attn_prev is None or attn_last is None or tokens_prev is None or tokens_last is None:
                return None

            grads_prev, grads_last = torch.autograd.grad(
                s,
                [tokens_prev, tokens_last],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

            bsz, _, tgt_len, src_len = attn_prev.shape
            if tgt_len != src_len:
                return None

            eye = torch.eye(tgt_len, device=attn_prev.device, dtype=attn_prev.dtype).unsqueeze(0).repeat(bsz, 1, 1)

            def _build_a(attn, cam_tokens, grads):
                token_weights = (cam_tokens.float() * grads.float()).sum(dim=-1)  # (B,T)
                token_weights = F.relu(token_weights)
                token_weights = token_weights / (token_weights.sum(dim=-1, keepdim=True) + 1e-8)

                a = attn.mean(dim=1)  # (B,T,T)
                a = a * token_weights.unsqueeze(1)  # weight source tokens
                a = a + eye
                a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                return a

            a_prev = _build_a(attn_prev, tokens_prev, grads_prev)
            a_last = _build_a(attn_last, tokens_last, grads_last)

            w_prev = 0.5
            w_last = 1.0
            a_avg = (a_prev * w_prev + a_last * w_last) / (w_prev + w_last + 1e-8)
            a_avg = a_avg / (a_avg.sum(dim=-1, keepdim=True) + 1e-8)

            j = torch.bmm(a_last, a_avg)
            cam = j[:, 0, 1:].view(bsz, grid, grid)

            if clip_quantile is not None:
                flat = cam.view(cam.size(0), -1)
                q = torch.quantile(flat, clip_quantile, dim=1, keepdim=True)
                cam = torch.minimum(cam, q.view(-1, 1, 1))

            cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
            cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + eps)

            h, w_img = base.shape[-2:]
            cam_up = F.interpolate(cam.unsqueeze(1), size=(h, w_img), mode="bilinear", align_corners=False)
            return cam_up.detach()
    finally:
        model.train(was_training)

def get_backbone_specific_attn_up(model, base, device, clip_quantile: float = 0.99, eps: float = 1e-8):
    """
    Unified attention-to-mask dispatcher.

    ViT path:
        keep existing get_gradrollout_last2avg_attn_up unchanged.

    RN50 / ResNet-style path:
        use model.get_rn50_singlelayer_gradrollout_top1 and convert to (B,1,H,W).
    """
    enc = model.image_encoder

    if hasattr(enc, "transformer"):
        return get_gradrollout_last2avg_attn_up(
            model, base, device, clip_quantile=clip_quantile, eps=eps
        )

    if hasattr(enc, "attnpool") and hasattr(model, "get_rn50_singlelayer_gradrollout_top1"):
        cam, _, _ = model.get_rn50_singlelayer_gradrollout_top1(
            base,
            clip_quantile=clip_quantile,
            eps=eps,
            upsample=True,
        )
        if cam is None:
            return None
        if cam.dim() == 3:
            cam = cam.unsqueeze(1)
        return cam.detach()

    return None

def get_backbone_specific_attn_up_batch(
    model,
    bases,
    device,
    clip_quantile: float = 0.99,
    eps: float = 1e-8,
):
    outs = []
    for i in range(bases.size(0)):
        attn_up_i = get_backbone_specific_attn_up(
            model,
            bases[i:i + 1],
            device,
            clip_quantile=clip_quantile,
            eps=eps,
        )
        if attn_up_i is None:
            return None
        outs.append(attn_up_i)
    return torch.cat(outs, dim=0)

def build_high_mid_low_masks(attn_up, p_high, p_low):
    """
    Build three masks based on quantiles.
    attn_up: (1,1,H,W)
    """
    flat = attn_up.view(-1)
    high_thr = torch.quantile(flat, 1 - p_high) if p_high > 0 else float('inf')
    low_thr = torch.quantile(flat, p_low) if p_low > 0 else float('-inf')
    # Hard masks with a deterministic tie-breaking rule:
    # high priority > low priority, mid is the remainder.
    if p_high > 0:
        high_bool = attn_up > high_thr
    else:
        high_bool = torch.zeros_like(attn_up, dtype=torch.bool)
    if p_low > 0:
        low_bool = (attn_up <= low_thr) & (~high_bool)
    else:
        low_bool = torch.zeros_like(attn_up, dtype=torch.bool)
    mid_bool = (~high_bool) & (~low_bool)
    M_high = high_bool.to(dtype=attn_up.dtype)
    M_low = low_bool.to(dtype=attn_up.dtype)
    M_mid = mid_bool.to(dtype=attn_up.dtype)
    return M_high, M_mid, M_low

def _clamp01(u, eps=1e-6):
    u = float(u)
    eps = float(eps)
    if eps < 0:
        eps = 0.0
    if eps >= 0.5:
        eps = 0.499999
    return float(min(max(u, eps), 1.0 - eps))

def map_energy_to_quantiles(E_tilde, qh_min=0.05, qh_max=0.95, ql_min=0.05, ql_max=0.95, gamma=2.0, q_eps=1e-6):
    """
    Deterministic quantile mapping (monotonic w.r.t. E_tilde in [0,1]).
    - For mh: u_h decreases as E increases (high entropy -> lower quantile -> smaller mh).
    - For ml: u_l increases as E increases (high entropy -> higher quantile -> larger ml).
    """
    E = float(E_tilde)
    E = float(min(max(E, 0.0), 1.0))
    g = float(gamma)
    if g <= 0:
        g = 1.0
    Eh = E ** g

    qh_min = float(qh_min); qh_max = float(qh_max)
    ql_min = float(ql_min); ql_max = float(ql_max)

    u_h = qh_max - (qh_max - qh_min) * Eh
    u_l = ql_min + (ql_max - ql_min) * Eh
    return _clamp01(u_h, q_eps), _clamp01(u_l, q_eps)

def inv_cdf_exp_pos_trunc(u, a, b, T):
    """
    Inverse-CDF for truncated exp family on [a,b] with density 鈭?exp(m/T).
    Uses numerically stable form:
      m = a + T * log(1 + u * (exp((b-a)/T) - 1))
    """
    u = _clamp01(u, 1e-12)
    a = float(a); b = float(b); T = float(T)
    if b < a:
        a, b = b, a
    if T <= 0:
        return b  # as T->0, distribution collapses to upper edge
    delta = (b - a) / T
    # log1p(u*(exp(delta)-1))
    m = a + T * math.log1p(u * (math.expm1(delta)))
    return float(min(max(m, a), b))

def inv_cdf_exp_neg_trunc(u, a, b, T):
    """
    Inverse-CDF for truncated exp family on [a,b] with density 鈭?exp(-m/T).
    Uses stable form:
      m = a - T * log(1 - u * (1 - exp(-(b-a)/T)))
    """
    u = _clamp01(u, 1e-12)
    a = float(a); b = float(b); T = float(T)
    if b < a:
        a, b = b, a
    if T <= 0:
        return a  # as T->0, distribution collapses to lower edge
    delta = (b - a) / T
    m = a - T * math.log1p(-u * (1.0 - math.exp(-delta)))
    return float(min(max(m, a), b))

def apply_attn_guided_mix(
    views,
    bases,
    mixes,
    betas,
    model,
    args,
):
    """
    Apply attention-guided local mix on a list of views (each: (3,H,W), cpu tensor).
    One-shot piecewise interpolation between per-view base and pure_mix guided by attention.
    - views:  baseline AugMix views v_j (only used for view0; kept for indexing compatibility)
    - bases:  per-view preaugment tensors b_j (spatially aligned to each view)
    - mixes:  per-view pure mix tensors mix_j (sum of augmentation chains before Beta mixing)
    - betas:  list of per-view Beta scalars m_j used by baseline AugMix (view0 placeholder included)
    """
    # True degeneration: return original AugMix views when attention is effectively off
    if (args.attn_p_high <= 0.0 and args.attn_p_low <= 0.0):
        return views

    if (args.attn_m_high == 0.0 and args.attn_m_low == 0.0):
        return views

    new_views = []
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    for idx, v in enumerate(views):
        # keep anchor (view0) untouched
        if idx == 0:
            new_views.append(v)
            continue

        base = bases[idx].unsqueeze(0).to(device, non_blocking=True) if bases is not None else v.unsqueeze(0).to(device, non_blocking=True)
        pure_mix = mixes[idx].unsqueeze(0).to(device, non_blocking=True) if mixes is not None else v.unsqueeze(0).to(device, non_blocking=True)
        beta_j = betas[idx] if betas is not None and idx < len(betas) else 1.0
        beta_j = float(beta_j)
        attn_up = get_backbone_specific_attn_up(model, base, device)
        if attn_up is None:
            new_views.append(v)
            continue
        M_high, M_mid, M_low = build_high_mid_low_masks(attn_up, args.attn_p_high, args.attn_p_low)

        m_high = float(args.attn_m_high)
        m_low = float(args.attn_m_low)

        beta_t = torch.tensor(beta_j, dtype=M_mid.dtype, device=M_mid.device)
        mh_t = torch.tensor(float(m_high), dtype=M_high.dtype, device=M_high.device)
        ml_t = torch.tensor(float(m_low), dtype=M_low.dtype, device=M_low.device)
        m_map = (M_high * mh_t) + (M_mid * beta_t) + (M_low * ml_t)  # (1,1,H,W)
        v_new = m_map * base + (1.0 - m_map) * pure_mix
        new_views.append(v_new.squeeze(0).cpu())
    return new_views

def test_time_tuning(model, inputs, optimizer, scaler, args):
    
    selected_idx = None
    for j in range(args.tta_steps):
        if True:
            output = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = entropy_avg(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def main():
    args = parser.parse_args()
    _setup_distributed(args)
    set_random_seed(args.seed)
        # Auto dataset-specific ctx_init (insert domain word between "a" and "photo of a")
    # Example: Caltech101 -> "a_People_photo_of_a" (5 tokens after "_"->" ").
    if args.ctx_init == "a_photo_of_a":
        _domain_word = {
            "caltech101": "object",
            "dtd": "texture",
            "aircraft": "aircraft",
            "flower102": "flower",
            "ucf101": "action",
            "cars": "car",
            "eurosat": "satellite-map",
            "pets": "pet",
        }.get(str(args.test_sets).lower())
        if _domain_word is not None:
            args.ctx_init = f"a_{_domain_word}_photo_of_a"
            # Keep args.n_ctx in sync for logging; PromptLearner will also infer n_ctx from ctx_init.
            args.n_ctx = len(args.ctx_init.replace("_", " ").split())
            if _is_main_process():
                print(f'=> Auto ctx_init: "{args.ctx_init}" (n_ctx={args.n_ctx})')
    if not (0.0 <= args.attn_p_high <= 1.0 and 0.0 <= args.attn_p_low <= 1.0):
        raise ValueError("attn_p_high and attn_p_low must be in [0,1].")
    if args.attn_p_high + args.attn_p_low > 1.0:
        raise ValueError("attn_p_high + attn_p_low must be <= 1.0.")

    args.alpha = args.eps / 4.0
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets, 'eps_'+str(args.eps)+'_alpha_'+str(args.alpha)+'_step_'+str(args.steps))

    os.makedirs(args.output_dir, exist_ok=True)

    def _open_unique_log_file(output_dir: str, filename: str = "log.txt"):
        base_path = os.path.join(output_dir, filename)
        if not os.path.exists(base_path):
            return open(base_path, "w"), base_path
        stem, ext = os.path.splitext(filename)
        k = 1
        while True:
            cand = os.path.join(output_dir, f"{stem}_{k}{ext}")
            if not os.path.exists(cand):
                return open(cand, "w"), cand
            k += 1

    if _is_main_process():
        args.out_file, log_path = _open_unique_log_file(args.output_dir, "log.txt")
    else:
        args.out_file = open(os.devnull, "w")
        log_path = os.devnull
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    if _is_main_process():
        print(f"=> Logging to: {log_path}")

    assert args.gpu is not None

    # make RNG streams differ across ranks to avoid accidental identical augmentation randomness
    set_random_seed(int(args.seed) + int(getattr(args, "rank", 0)))
    if _is_main_process():
        print("Use GPU: {} for training".format(args.gpu))

    # model
    dset = args.test_sets
    if len(dset) > 1: 
        classnames = eval("{}_classes".format(dset.lower()))
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    args.classnames = classnames

    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    model.rn50_cam_sharpen = args.rn50_cam_sharpen
    model_state = None

    ###### load robust vision encoder (TeCoA) ######
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            'RN50-eps1': 'pretrain/tecoa/rn50_eps1.pth.tar',
        }[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
        model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
        if _is_main_process():
            print('load robust vision encoder')

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
                param.requires_grad_(False)

    if _is_main_process():
        print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        if _is_main_process():
            print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    scaler = None
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    
    results = {}
    if True:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            ])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(dset)>1, return_base=args.view_gen_mode=='attn_augmix')
        batchsize = 1

        val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
        if _is_main_process():
            print("number of test samples: {}".format(len(val_dataset)))

        sampler = None
        if getattr(args, "distributed", False):
            sampler = _DistributedEvalSampler(val_dataset)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize,
            shuffle=False if sampler is not None else False,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
        )

        if _is_main_process():
            print("evaluating: {}".format(dset))
        
        results = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform)
        del val_dataset, val_loader
        if args.eps <= 0:
            print_log = "=> Acc. on testset [{}]: Clean Acc @1 {}/ TTA Clean Acc @1 {}".format(dset, results[0], results[1])
            save_log = {'clean_acc': results[0], 'tta_clean_acc': results[1]}
        else:
            print_log = "=> Acc. on testset [{}]: Adv Acc @1 {}/ TTA Adv Acc @1 {} ".format(dset, results[0], results[1])
            save_log = {'adv_acc': results[0], 'tta_adv_acc': results[1]}
      
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        if _is_main_process():
            print(print_log+'\n')

        if _is_main_process():
            torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = None
    if _is_main_process():
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, tpt1],
            prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    if args.eps > 0.0:
        assert args.steps > 0
        atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)


        # flatten possible nested lists from DataLoader collate (batch_size=1)
        while isinstance(images, list) and len(images) == 1 and isinstance(images[0], (list, tuple)):
            images = images[0]

        # unpack transform outputs (views, bases, mixes, betas) if provided
        bases = None
        mixes = None
        betas = None
        if isinstance(images, (list, tuple)) and len(images) == 2:
            images, bases = images
        elif isinstance(images, (list, tuple)) and len(images) == 4:
            images, bases, mixes, betas = images
        if isinstance(bases, list) and len(bases) == 1 and isinstance(bases[0], list):
            bases = bases[0]
        if isinstance(mixes, list) and len(mixes) == 1 and isinstance(mixes[0], list):
            mixes = mixes[0]

        # generate views (AugMix baseline)
        if args.eps > 0.0:
            # anchor for PGD (use first view)
            anchor = images
            # unwrap nested list/tuple to get tensor
            while isinstance(anchor, (list, tuple)):
                anchor = anchor[0]
            if not isinstance(anchor, torch.Tensor):
                raise ValueError("Unexpected image type for PGD attack.")
            anchor = anchor.cuda(args.gpu, non_blocking=True)
            adv_image = atk(anchor, target)
            img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
            images = generate_augmix_views_baseline(img_adv, data_transform)
            # regenerate bases if needed
            if isinstance(images, (list, tuple)) and len(images) == 2:
                images, bases = images
                mixes, betas = None, None
            elif isinstance(images, (list, tuple)) and len(images) == 4:
                images, bases, mixes, betas = images
        else:
            if not isinstance(images, list):
                if len(images.size()) > 4:
                    images = images.squeeze(0)
                images = [images]
            if bases is None and args.view_gen_mode == 'attn_augmix':
                # best-effort: use the same views as bases when transform didn't return bases
                bases = images
            if mixes is None and args.view_gen_mode == 'attn_augmix':
                mixes = images
            if betas is None and args.view_gen_mode == 'attn_augmix':
                betas = [1.0 for _ in range(len(images))]

        # standardize to list of (3,H,W)
        images = [img.squeeze(0) if img.dim() == 4 else img for img in images]
        if bases is not None:
            bases = [b.squeeze(0) if b.dim() == 4 else b for b in bases]
        if mixes is not None:
            mixes = [m.squeeze(0) if hasattr(m, "dim") and m.dim() == 4 else m for m in mixes]

        # optional attention-guided mix
        if args.view_gen_mode == 'attn_augmix':
            images = apply_attn_guided_mix(images, bases, mixes, betas, model, args)
        elif args.view_gen_mode != 'augmix':
            raise ValueError(f"Unknown view_gen_mode: {args.view_gen_mode}")

        # move to device and stack
        images = [img.cuda(args.gpu, non_blocking=True) for img in images]
        images = torch.stack(images, dim=0)  # (B,3,H,W)
        image = images[:1]  # first view
                # Reset model/optimizer state per sample to avoid leaking tuned prompts across samples.
                
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        with torch.no_grad():
            clip_output = model(image)
            clip_features, _, _ = model.forward_features(images)

        assert args.tta_steps > 0
        test_time_tuning(model, images, optimizer, scaler, args)
        with torch.no_grad():
            tuned_outputs = model(images)
        
        # Select top views by tuned predictive entropy for reliability fusion
        selected_v = int(getattr(args, "selected_v", images.size(0)))
        if selected_v <= 0 or selected_v > images.size(0):
            selected_v = images.size(0)
        tuned_entropy = -(tuned_outputs.softmax(1) * tuned_outputs.log_softmax(1)).sum(1)  # (B,)
        selected_idx = torch.argsort(tuned_entropy, descending=False)[:selected_v]  # (selected_v,)
        selected_features = clip_features[selected_idx]  # (selected_v, d)
        selected_outputs = tuned_outputs[selected_idx]   # (selected_v, C)

        sim_matrix_images = torch.bmm(selected_features.unsqueeze(0), selected_features.unsqueeze(0).permute(0, 2, 1))
        score = get_top_sim(sim_matrix_images)
        weight = torch.nn.functional.softmax(score/0.01, dim=-1)
        tta_output = torch.bmm(weight.unsqueeze(-1).transpose(1, 2), selected_outputs.unsqueeze(0)).squeeze(1)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))
       
        top1.update(acc1[0], images.size(0))
        tpt1.update(tpt_acc1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if _is_main_process() and ((i+1) % args.print_freq == 0 or (i+1) == len(val_loader)):
            if args.eps <= 0:
                print_log = 'iter:{}/{}, clip_acc1={}, tta_acc1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            else:
                print_log = 'iter:{}/{}, clip_adv1={}, tta_adv1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            progress.display(i)

    # reduce meters across ranks (sum/count) and recompute avg
    if getattr(args, "distributed", False) and _dist_is_avail_and_initialized():
        for meter in (top1, tpt1):
            t = torch.tensor([float(meter.sum), float(meter.count)], device=f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            meter.sum = float(t[0].item())
            meter.count = float(t[1].item())
            meter.avg = meter.sum / meter.count if meter.count > 0 else 0.0

    if _is_main_process() and progress is not None:
        progress.display_summary()

    return [top1.avg, tpt1.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/atpt')

    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--steps', type=int, default=0)

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps') 
    #退化为单步TTA时设置为1

    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'])
    parser.add_argument('--view-gen-mode', type=str, default='attn_augmix', choices=['augmix', 'attn_augmix'], help='multi-view generation mode')
    parser.add_argument('--attn_p_high', type=float, default=0.2, help='high-attention area ratio')
    parser.add_argument('--attn_p_low', type=float, default=0.8, help='low-attention area ratio')
    parser.add_argument('--attn_m_high', type=float, default=0.8, help='mix coefficient for high-attention area')
    parser.add_argument('--attn_m_low', type=float, default=0.2, help='mix coefficient for low-attention area')
    parser.add_argument('--selected_v', type=int, default=64, help='number of lowest-entropy tuned views used for reliability fusion (default: 22)') 
    #退化为使用全部视图时设置为64
    parser.add_argument('--rn50_cam_sharpen', type=float, default=0.2, help='RN50 Grad-CAM support tightening threshold') 
    #退化为不使用sharpen时设置为0.0

    main()
