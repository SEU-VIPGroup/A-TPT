# Official implementation for **A-TPT**

## 1. Framework
![image](fm.png)

Our implementation follows previous works [RTPT](https://github.com/TomSheng21/R-TPT) and [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch). 
Thanks to their authors for providing reprodible experimental results and detailed code basis. 
Based on prompt tuning implementation in [TPT](https://github.com/azshue/TPT) and [RTPT](https://github.com/azshue/TPT), we introduce attention-guided process towards fine-graind robustness.


## 2. Environment Setup

Recommended:

- Python 3.10+
- pytorch == 1.12.1
- torchvision == 0.13.1

Install minimal dependencies:

```bash
pip install torch torchvision
pip install numpy pillow tqdm ftfy regex torchattacks
```

Notes:

- The repo uses local `clip/` and `torchattacks/` modules.
- CLIP checkpoints are cached under `cache/clip/`.

## 3. Dataset

We follow [TPT](https://github.com/azshue/TPT) dataset layout. Please download required datasets and check the path of json file.
- `I` -> `imagenet/images` (reads `val/`)
- `Caltech101`, `DTD`, `Flower102`, `Food101`, `Cars`, `SUN397`, `Aircraft`, `Pets`, `UCF101`, `eurosat` use few-shot style splits with json files

Example layout:
```bash
/path/to/your/dataset
    └─ dtd\
       ├─ images\
       │  ├─ banded\
       │  │  ├─ banded_0002.jpg
       │  │  └─ ...
       │  ├─ blotchy\
       │  └─ ... 
       ├─ dtd\
       │  └─ split_zhou_DescribableTextures.json
       ├─ labels\        
       └─ imdb\         
```

## 4. Quick Start

### 4.1 Clean evaluation

```bash
python atpt.py dataset --test_sets DTD --dataset_mode test -a ViT-B/32 -p 50 --ctx_init a_photo_of_a --seed 0 --output_dir output_results/ckps/rtpt --eps 0.0 --view-gen-mode attn_augmix --attn_p_high 0.2 --attn_p_low 0.8 --attn_m_high 0.8 --attn_m_low 0.2
```

### 4.2 Adversarial evaluation 
```bash
python atpt.py dataset --test_sets DTD --dataset_mode test -a ViT-B/32 -p 50 --ctx_init a_photo_of_a --seed 0 --output_dir output_results/ckps/rtpt --eps 4.0 --steps 100 --view-gen-mode attn_augmix --attn_p_high 0.2 --attn_p_low 0.8 --attn_m_high 0.8 --attn_m_low 0.2
```
