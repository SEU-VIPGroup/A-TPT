
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .model import LayerNorm
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

import copy

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='cache/clip'

mu = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)


class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input):
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding[:40]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class RN50SingleLayerSelfAttentionAnalysisHead(nn.Module):
    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.ln_pre = LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads)
        self.ln_post = LayerNorm(width)

    def init_from_attnpool(self, attnpool: nn.Module) -> None:
        width = attnpool.q_proj.in_features
        with torch.no_grad():
            self.attn.in_proj_weight[:width].copy_(attnpool.q_proj.weight)
            self.attn.in_proj_weight[width:2 * width].copy_(attnpool.k_proj.weight)
            self.attn.in_proj_weight[2 * width:].copy_(attnpool.v_proj.weight)
            self.attn.in_proj_bias[:width].copy_(attnpool.q_proj.bias)
            self.attn.in_proj_bias[width:2 * width].copy_(attnpool.k_proj.bias)
            self.attn.in_proj_bias[2 * width:].copy_(attnpool.v_proj.bias)
            nn.init.eye_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward_with_attn_from_feature_map(
        self,
        feat_map: torch.Tensor,
        positional_embedding: torch.Tensor,
    ):
        bsz, channels, grid, grid_w = feat_map.shape
        if grid != grid_w:
            raise RuntimeError("RN50 analysis head expects square feature maps.")

        x = feat_map.flatten(start_dim=2).permute(0, 2, 1)
        cls = x.mean(dim=1, keepdim=True)
        x = torch.cat([cls, x], dim=1)

        if positional_embedding.shape[0] != (grid * grid + 1):
            raise RuntimeError("RN50 analysis head positional embedding shape mismatch.")
        x = x + positional_embedding.to(device=x.device, dtype=x.dtype).unsqueeze(0)

        cam_tokens = x

        x_norm = self.ln_pre(x)
        x_lnd = x_norm.permute(1, 0, 2)
        attn_out, attn_map = self.attn(
            x_lnd,
            x_lnd,
            x_lnd,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attn_out.permute(1, 0, 2)

        cls_token = self.ln_post(x[:, 0, :])
        return cls_token, [attn_map], [cam_tokens], grid


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, ):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion

        self.normalize = ImageNormalizer(mu, std).cuda(device)
        # analysis-only head for RN50 mask computation; normal CLIP inference stays unchanged.
        self.rn50_analysis_head = None
        if hasattr(self.image_encoder, "attnpool") and not hasattr(self.image_encoder, "transformer"):
            width = self.image_encoder.attnpool.q_proj.in_features
            heads = self.image_encoder.attnpool.num_heads
            self.rn50_analysis_head = RN50SingleLayerSelfAttentionAnalysisHead(width, heads)
            self.rn50_analysis_head.init_from_attnpool(self.image_encoder.attnpool)
            self.rn50_analysis_head.to(
                device=self.image_encoder.attnpool.q_proj.weight.device,
                dtype=self.image_encoder.attnpool.q_proj.weight.dtype,
            )
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        
        image_features = self.image_encoder(self.normalize(image.type(self.dtype)))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
       
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)
        
    def forward_features(self, input):
        image_features = self.image_encoder(self.normalize(input.type(self.dtype)))

        text_features = self.get_text_features()       

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features, logit_scale

    def _forward_resnet_to_layer4(self, normalized: torch.Tensor) -> torch.Tensor:
        enc = self.image_encoder

        def stem(x):
            x = enc.relu1(enc.bn1(enc.conv1(x)))
            x = enc.relu2(enc.bn2(enc.conv2(x)))
            x = enc.relu3(enc.bn3(enc.conv3(x)))
            x = enc.avgpool(x)
            return x

        x = normalized.type(enc.conv1.weight.dtype)
        x = stem(x)
        x = enc.layer1(x)
        x = enc.layer2(x)
        x = enc.layer3(x)
        x = enc.layer4(x)
        return x

    def get_attention_map(self, input):
        """
        Return attention map from the visual encoder for a single batch of images.
        Supports both ResNet-style and ViT-style CLIP vision encoders.
        """
        normalized = self.normalize(input.type(self.dtype))
        attn_info = None
        if hasattr(self.image_encoder, "forward_with_attn"):
            outputs = self.image_encoder.forward_with_attn(normalized)
            # ModifiedResNet returns (features, attn, grid), ViT returns the same tuple
            if isinstance(outputs, Tuple) and len(outputs) in (3, 4):
                if len(outputs) == 3:
                    _, attn, grid_hint = outputs
                else:
                    _, attn, _, grid_hint = outputs
                if isinstance(attn, (list, tuple)):
                    attn = attn[-1]
                if attn is not None:
                    # attn: (batch, heads, tgt_len, src_len)
                    # CLS 对所有 patch 的注意力
                    spatial = attn[:, :, 0, 1:]
                    # 归一化，避免丢弃 CLS 后残差主导
                    spatial = spatial / (spatial.sum(dim=-1, keepdim=True) + 1e-8)
                    # 动态推断网格，避免分辨率/patch 不匹配造成错位
                    tokens = spatial.size(-1)
                    grid = int((tokens) ** 0.5)
                    if grid * grid != tokens:
                        grid = grid_hint
                    spatial = spatial.reshape(spatial.size(0), spatial.size(1), grid, grid)
                    attn_info = {
                        "per_head": spatial.detach(),
                        "avg": spatial.mean(dim=1).detach(),
                        "grid": grid
                    }
        return attn_info

    def get_gradcam_top1(self, images, clip_quantile: float = 0.99, eps: float = 1e-8, upsample: bool = True):
        """Single-layer token Grad-CAM (top-1 logit) for ViT-based CLIP.

        Uses the last Transformer block's input tokens (cam_tokens_all[-1]) and its gradient.

        Returns: cam, logits, top1
          - cam: (B,H,W) if upsample else (B,grid,grid)
          - logits: (B,C) detached
          - top1: (B,) indices
        """
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                normalized = self.normalize(images.type(self.dtype))
                outputs = self.image_encoder.forward_with_attn(normalized)
                if not isinstance(outputs, Tuple) or len(outputs) != 4:
                    raise RuntimeError(
                        "get_gradcam_top1 expects ViT forward_with_attn to return (features, attn_maps_all, cam_tokens_all, grid_size)."
                    )

                features, _, cam_tokens_all, grid = outputs
                if not isinstance(cam_tokens_all, (list, tuple)) or len(cam_tokens_all) == 0:
                    raise RuntimeError("cam_tokens_all is empty")

                cam_tokens = cam_tokens_all[-1]

                features = features / features.norm(dim=-1, keepdim=True)
                text_features = self.get_text_features()
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * features @ text_features.t()  # (B,C)

                top1 = logits.detach().argmax(dim=-1)
                s = logits[torch.arange(logits.size(0), device=logits.device), top1].sum()

                self.zero_grad(set_to_none=True)
                s.backward()

                grads = cam_tokens.grad
                if grads is None:
                    raise RuntimeError("cam_tokens.grad is None (did you call under no_grad/inference_mode?)")

                p = cam_tokens[:, 1:, :].float()  # (B,N,D)
                g = grads[:, 1:, :].float()       # (B,N,D)
                w = g.mean(dim=1)                 # (B,D)
                m = F.relu((p * w[:, None, :]).sum(dim=-1))  # (B,N)
                cam = m.view(m.size(0), grid, grid)

                if clip_quantile is not None:
                    flat = cam.view(cam.size(0), -1)
                    q = torch.quantile(flat, clip_quantile, dim=1, keepdim=True)
                    cam = torch.minimum(cam, q.view(-1, 1, 1))

                cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
                cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
                cam = (cam - cam_min) / (cam_max - cam_min + eps)

                if upsample:
                    h, w_img = images.shape[-2:]
                    cam = F.interpolate(cam.unsqueeze(1), size=(h, w_img), mode="bilinear", align_corners=False).squeeze(1)

                return cam, logits.detach(), top1
        finally:
            self.train(was_training)


    def get_gradrollout_top1(self, images, clip_quantile: float = 0.99, eps: float = 1e-8, upsample: bool = True):
        """Token Grad-CAM (top-1 logit) for ViT-based CLIP visual encoder.

        Returns: cam, logits, top1
          - cam: (B,H,W) if upsample else (B,grid,grid)
          - logits: (B,C) detached
          - top1: (B,) indices
        """
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                normalized = self.normalize(images.type(self.dtype))
                outputs = self.image_encoder.forward_with_attn(normalized)
                if not isinstance(outputs, Tuple) or len(outputs) != 4:
                    raise RuntimeError("get_gradcam_top1 expects ViT forward_with_attn to return (features, attn_maps_all, cam_tokens_all, grid_size).")

                features, attn_maps_all, cam_tokens_all, grid = outputs
                if not isinstance(attn_maps_all, (list, tuple)) or len(attn_maps_all) == 0:
                    raise RuntimeError("attn_maps_all is empty; cannot compute Grad-Rollout")

                features = features / features.norm(dim=-1, keepdim=True)
                text_features = self.get_text_features()
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * features @ text_features.t()  # (B,C)

                top1 = logits.detach().argmax(dim=-1)
                s = logits[torch.arange(logits.size(0), device=logits.device), top1].sum()

                self.zero_grad(set_to_none=True)
                s.backward()

                # Grad-Rollout: cross-layer fusion (rollout) using per-layer token Grad-CAM weights
                first = attn_maps_all[0]
                if first is None:
                    raise RuntimeError("attn_maps_all[0] is None")
                bsz, n_heads, tgt_len, src_len = first.shape
                if tgt_len != src_len:
                    raise RuntimeError("Expected square attention matrices")

                eye = torch.eye(tgt_len, device=first.device, dtype=first.dtype).unsqueeze(0).repeat(bsz, 1, 1)
                j = eye

                for attn, cam_tokens in zip(attn_maps_all, cam_tokens_all):
                    if attn is None or cam_tokens is None:
                        continue

                    grads = cam_tokens.grad
                    if grads is None:
                        raise RuntimeError("cam_tokens.grad is None; ensure retain_grad() is set and gradients are enabled")

                    token_weights = (cam_tokens.float() * grads.float()).sum(dim=-1)  # (B,T)
                    token_weights = F.relu(token_weights)
                    token_weights = token_weights / (token_weights.sum(dim=-1, keepdim=True) + 1e-8)

                    a = attn.mean(dim=1)  # (B,T,T)
                    a = a * token_weights.unsqueeze(1)  # weight source tokens
                    a = a + eye
                    a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                    j = torch.bmm(a, j)

                cam = j[:, 0, 1:].view(bsz, grid, grid)

                if clip_quantile is not None:
                    flat = cam.view(cam.size(0), -1)
                    q = torch.quantile(flat, clip_quantile, dim=1, keepdim=True)
                    cam = torch.minimum(cam, q.view(-1, 1, 1))

                cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
                cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
                cam = (cam - cam_min) / (cam_max - cam_min + eps)

                if upsample:
                    h, w_img = images.shape[-2:]
                    cam = F.interpolate(cam.unsqueeze(1), size=(h, w_img), mode="bilinear", align_corners=False).squeeze(1)

                return cam, logits.detach(), top1
        finally:
            self.train(was_training)

    def get_gradrollout_weight_top1(self, images, clip_quantile: float = 0.99, eps: float = 1e-8, upsample: bool = True):
        """Grad-Rollout with depth-increasing layer weights (top-1 logit) for ViT-based CLIP.

        Compared to `get_gradrollout_top1`, this method applies a monotonically increasing
        scalar weight to deeper layers so later layers contribute more in the rollout.

        Returns: cam, logits, top1
          - cam: (B,H,W) if upsample else (B,grid,grid)
          - logits: (B,C) detached
          - top1: (B,) indices
        """
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                normalized = self.normalize(images.type(self.dtype))
                outputs = self.image_encoder.forward_with_attn(normalized)
                if not isinstance(outputs, Tuple) or len(outputs) != 4:
                    raise RuntimeError(
                        "get_gradrollout_weight_top1 expects ViT forward_with_attn to return (features, attn_maps_all, cam_tokens_all, grid_size)."
                    )

                features, attn_maps_all, cam_tokens_all, grid = outputs
                if not isinstance(attn_maps_all, (list, tuple)) or len(attn_maps_all) == 0:
                    raise RuntimeError("attn_maps_all is empty; cannot compute Grad-Rollout")

                features = features / features.norm(dim=-1, keepdim=True)
                text_features = self.get_text_features()
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * features @ text_features.t()  # (B,C)

                top1 = logits.detach().argmax(dim=-1)
                s = logits[torch.arange(logits.size(0), device=logits.device), top1].sum()

                self.zero_grad(set_to_none=True)
                s.backward()

                first = attn_maps_all[0]
                if first is None:
                    raise RuntimeError("attn_maps_all[0] is None")
                bsz, _, tgt_len, src_len = first.shape
                if tgt_len != src_len:
                    raise RuntimeError("Expected square attention matrices")

                eye = torch.eye(tgt_len, device=first.device, dtype=first.dtype).unsqueeze(0).repeat(bsz, 1, 1)
                j = eye

                num_layers = len(attn_maps_all)
                layer_weights = torch.linspace(1.0, 2.0, num_layers, device=first.device, dtype=torch.float32)
                layer_weights = layer_weights / layer_weights.max()  # map to [0.5, 1.0]

                for layer_idx, (attn, cam_tokens) in enumerate(zip(attn_maps_all, cam_tokens_all)):
                    if attn is None or cam_tokens is None:
                        continue

                    grads = cam_tokens.grad
                    if grads is None:
                        raise RuntimeError("cam_tokens.grad is None; ensure retain_grad() is set and gradients are enabled")

                    token_scores = (cam_tokens.float() * grads.float()).sum(dim=-1)  # (B,T)
                    token_scores = F.relu(token_scores)
                    token_scores = token_scores * layer_weights[layer_idx]
                    token_weights = token_scores / (token_scores.sum(dim=-1, keepdim=True) + 1e-8)

                    a = attn.mean(dim=1)  # (B,T,T)
                    a = a * token_weights.unsqueeze(1)  # weight source tokens

                    a = a * layer_weights[layer_idx]  # depth-increasing layer weight
                    a = a + eye
                    a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                    j = torch.bmm(a, j)

                cam = j[:, 0, 1:].view(bsz, grid, grid)

                if clip_quantile is not None:
                    flat = cam.view(cam.size(0), -1)
                    q = torch.quantile(flat, clip_quantile, dim=1, keepdim=True)
                    cam = torch.minimum(cam, q.view(-1, 1, 1))

                cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
                cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
                cam = (cam - cam_min) / (cam_max - cam_min + eps)

                if upsample:
                    h, w_img = images.shape[-2:]
                    cam = F.interpolate(cam.unsqueeze(1), size=(h, w_img), mode="bilinear", align_corners=False).squeeze(1)

                return cam, logits.detach(), top1
        finally:
            self.train(was_training)

    def get_gradrollout_last2avg_top1(self, images, clip_quantile: float = 0.99, eps: float = 1e-8, upsample: bool = True):
        """Grad-Rollout using only the last two layers with a weighted average fusion.

        Uses only the last two layers' (attn_map, cam_tokens) and:
        1) builds per-layer contribution matrices A_{L-1}, A_L
        2) forms a weighted average A_avg = (0.5*A_{L-1} + 1.0*A_L) / 1.5
        3) fuses with matrix multiplication J = A_L @ A_avg

        Returns: cam, logits, top1
          - cam: (B,H,W) if upsample else (B,grid,grid)
          - logits: (B,C) detached
          - top1: (B,) indices
        """
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                normalized = self.normalize(images.type(self.dtype))
                outputs = self.image_encoder.forward_with_attn(normalized)
                if not isinstance(outputs, Tuple) or len(outputs) != 4:
                    raise RuntimeError(
                        "get_gradrollout_last2avg_top1 expects ViT forward_with_attn to return (features, attn_maps_all, cam_tokens_all, grid_size)."
                    )

                features, attn_maps_all, cam_tokens_all, grid = outputs
                if not isinstance(attn_maps_all, (list, tuple)) or len(attn_maps_all) < 2:
                    raise RuntimeError("Need at least 2 layers for last-two-layer Grad-Rollout")

                features = features / features.norm(dim=-1, keepdim=True)
                text_features = self.get_text_features()
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * features @ text_features.t()  # (B,C)

                top1 = logits.detach().argmax(dim=-1)
                s = logits[torch.arange(logits.size(0), device=logits.device), top1].sum()

                self.zero_grad(set_to_none=True)
                s.backward()

                attn_last2 = attn_maps_all[-2:]
                tokens_last2 = cam_tokens_all[-2:]

                first = attn_last2[0]
                if first is None:
                    raise RuntimeError("attn_maps_all[-2] is None")
                bsz, _, tgt_len, src_len = first.shape
                if tgt_len != src_len:
                    raise RuntimeError("Expected square attention matrices")

                eye = torch.eye(tgt_len, device=first.device, dtype=first.dtype).unsqueeze(0).repeat(bsz, 1, 1)

                def build_a(attn, cam_tokens):
                    if attn is None or cam_tokens is None:
                        raise RuntimeError("attn/cam_tokens is None in last-two-layer rollout")
                    grads = cam_tokens.grad
                    if grads is None:
                        raise RuntimeError("cam_tokens.grad is None; ensure retain_grad() is set and gradients are enabled")

                    token_weights = (cam_tokens.float() * grads.float()).sum(dim=-1)  # (B,T)
                    token_weights = F.relu(token_weights)
                    token_weights = token_weights / (token_weights.sum(dim=-1, keepdim=True) + 1e-8)

                    a = attn.mean(dim=1)  # (B,T,T)
                    a = a * token_weights.unsqueeze(1)  # weight source tokens
                    a = a + eye
                    a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                    return a

                a_prev = build_a(attn_last2[0], tokens_last2[0])
                a_last = build_a(attn_last2[1], tokens_last2[1])

                w = torch.tensor([0.5, 1.0], device=first.device, dtype=torch.float32)
                a_avg = (a_prev * w[0] + a_last * w[1]) / (w.sum() + 1e-8)
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

                if upsample:
                    h, w_img = images.shape[-2:]
                    cam = F.interpolate(cam.unsqueeze(1), size=(h, w_img), mode="bilinear", align_corners=False).squeeze(1)

                return cam, logits.detach(), top1
        finally:
            self.train(was_training)

    def get_rn50_singlelayer_gradrollout_top1(
        self,
        images,
        clip_quantile: float = 0.99,
        eps: float = 1e-8,
        upsample: bool = True,
        text_features: torch.Tensor = None,
        logit_scale: torch.Tensor = None,
    ):
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                normalized = self.normalize(images.type(self.dtype))
                if not normalized.requires_grad:
                    normalized = normalized.detach().requires_grad_(True)

                feat_map = self._forward_resnet_to_layer4(normalized)

                if text_features is None:
                    with torch.no_grad():
                        text_features = self.get_text_features()

                if logit_scale is None:
                    with torch.no_grad():
                        logit_scale = self.logit_scale.exp()

                orig_features = self.image_encoder.attnpool(feat_map)
                orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
                orig_logits = logit_scale * orig_features @ text_features.t()
                top1 = orig_logits.detach().argmax(dim=-1)
                s = orig_logits[
                    torch.arange(orig_logits.size(0), device=orig_logits.device),
                    top1,
                ].sum()

                grads = torch.autograd.grad(
                    s,
                    feat_map,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                weights = grads.float().mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * feat_map.float()).sum(dim=1))

                if clip_quantile is not None:
                    flat = cam.view(cam.size(0), -1)
                    q = torch.quantile(flat, clip_quantile, dim=1, keepdim=True)
                    cam = torch.minimum(cam, q.view(-1, 1, 1))

                cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
                cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
                cam = (cam - cam_min) / (cam_max - cam_min + eps)

                if upsample:
                    h, w_img = images.shape[-2:]
                    cam = F.interpolate(
                        cam.unsqueeze(1),
                        size=(h, w_img),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

                # Tighten the returned RN50 heatmap support before atpt quantile masks
                # consume it, so the final mask focuses on a smaller activated region.
                cam_sharpen = float(getattr(self, "rn50_cam_sharpen", 0.2))
                cam = F.relu(cam - cam_sharpen)
                cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
                cam = cam / (cam_max + eps)
                cam = cam.pow(2.0)

                return cam, orig_logits.detach(), top1
        finally:
            self.train(was_training)

def get_coop(clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model
