<h1 align="center">🧞‍♀️ Jasmine: A simple, performant and scalable JAX-based world modeling codebase 🧞‍♀️</h1>

<p align="center">
    <a href= "https://github.com/FLAIROx/jafar/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

Jasmine is a production-ready JAX-based world modeling codebase. It currently implements the high-level architecture of [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) (Bruce et al., 2024) with [MaskGIT](https://arxiv.org/abs/2202.04200) (Chang et al., 2022), a diffusion baseline, as well as an autoregressive (causal) baseline. 

Jasmine scales from single hosts to hundreds of xPUs thanks to XLA and strives to be an easily hackable, batteries-included foundation for world modeling research.

<h2 name="overview" id="overview">Overview</h2>

Genie has three components: A [video tokenizer](jasmine/models/tokenizer.py), a [latent action model (LAM)](jasmine/models/lam.py), and a [dynamics model](jasmine/models/dynamics.py). 
The tokenizer needs to be trained first, which the dynamics then uses. Jasmine supports co-training the LAM with the dynamics model.
Jasmine implements a VQ-VAE and a MAE based tokenizer. The MaskGIT and causal baseline need to be trained using the discrete VQ-VAE tokenizer. The diffusion baseline is trained on latents from the MAE tokenizer.  The diffusion baseline, uses several elements from [Dreamer 4](https://arxiv.org/abs/2509.24527) (Hafner et. al., 2025) and is trained on the [diffusion forcing](https://arxiv.org/abs/2407.01392) (Chen et. al., 2024) objective.
The baselines and their respective training scripts can be found under `jasmine/baselines`.
```
jasmine
├── data
│   ├── jasmine_data
│   ├── pyproject.toml
├── jasmine
│   ├── baselines
│   │   ├── diffusion
│   │   │   ├── sample_diffusion.py
│   │   │   ├── train_dynamics_diffusion.py
│   │   │   └── train_tokenizer_mae.py
│   │   ├── maskgit
│   │   │   ├── sample_maskgit.py
│   │   │   ├── train_dynamics_causal.py
│   │   │   ├── train_dynamics_maskgit.py
|   │   │   └── train_tokenizer_vqvae.py
│   │   └── train_lam.py
│   ├── models
│   │   ├── dynamics.py
│   │   ├── genie.py
│   │   └── lam.py
│   └── utils
│       ├── dataloader.py
│       ├── nn.py
│       ├── preprocess.py
│       └── train_utils.py
├── LICENSE
├── pyproject.toml
└── README.md
```

<h2 name="overview" id="overview">Features</h2>

- Asynchronous & distributed checkpointing thanks to [orbax.checkpoint](https://github.com/google/orbax)
    - Jasmine also supports mixing and matching hardware topologies (e.g. train on four nodes, load the checkpoint on a single node)
- Optimized dataloading thanks to [Grain](https://github.com/google/grain)
    - Dataloading scales with the number of processes (i.e. nodes/xPUs)
- Checkpointing of model weights, optimizer and dataloader states
- Full reproducibility with **identical** training curves (thanks to seeded dataloading and training, and [JAX' approach to pseudo random numbers](https://docs.jax.dev/en/latest/random-numbers.html))
- Automatic checkpoint deletion/retention according to specified retention policy thanks to `orbax.checkpoint.CheckpointManager`
- Mixed precision training using `bfloat16`
    - `int8` training is on the roadmap via [aqt](https://github.com/google/aqt)
- FlashAttention thanks to [cuDNN SDPA](https://github.com/jax-ml/jax/blob/a155c5a9997924170e0067d552351a9833c12c11/jax/_src/cudnn/fused_attention_stablehlo.py#L842)
- Frame-level KV cache resets for accelerated spatiotemporal attention in causal baseline (still in PR)
- Activation checkpointing (even onto host memory if desired)
- DDP (changing to FSDP requires changing **a single line of code**)
- WSD learning rate schedule
    -  No need to retrain from scratch if you want to train for longer
- Index-shuffling during dataloading
- Google-native stack
    - https://github.com/google/orbax for checkpointing
    - https://github.com/google/grain for dataloading
    - https://github.com/google-deepmind/dm_pix for image manipulation
    - https://github.com/google/array_record as the data format
- Easy model inspection thanks to [treescope](https://github.com/google-deepmind/treescope)
- Modularized training script for easy inspection using notebooks ([demo notebook](https://colab.research.google.com/drive/1zHkciFIZxXloJgue9F5LtFlA0m00rJIf?usp=sharing))
- Easy model surgery thanks to the new [flax.nnx](https://flax.readthedocs.io/en/latest/migrating/linen_to_nnx.html) API
- [Shape suffixes](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd) throughout the repository

<h2 name="start" id="start">Setup 🧗</h2>

Jasmine requires `python 3.11`, `jax 0.7.2`, and `flax 0.11.2`. To install the requirements, run:

```bash
uv sync
pre-commit install
```

---

<h2 name="dataset" id="dataset">Dataset 📂</h2>

You can either download our preprocessed dataset from [Hugging Face](https://huggingface.co/datasets/p-doom/open_ai_minecraft_arrayrecords_chunked) or preprocess [OpenAI's VPT dataset](https://github.com/openai/Video-Pre-Training) manually.

### Option 1: Use Preprocessed Dataset (Recommended)

The easiest way to get started is to download our preprocessed dataset from Hugging Face. This script will handle downloading and extracting it:

```bash
bash data/jasmine_data/minecraft/huggingface/download_openai_array_records.sh
```

---

### Option 2: Manual Download & Preprocessing of OpenAI's VPT Dataset

If you prefer to use the raw VPT dataset from OpenAI and preprocess it yourself, follow these steps:

1. **Download index files:**
   This will download the initial index file:

   ```bash
   bash data/minecraft/openai/download_index_files.sh
   ```

2. **Download from all index files:**
   This may take a long time depending on your bandwidth:

   ```bash
   cd data/
   uv run python jasmine_data/minecraft/openai/download_videos.py --index_file_path data/open_ai_index_files/all_7xx_Apr_6.json
   uv run python jasmine_data/minecraft/openai/download_videos.py --index_file_path data/open_ai_index_files/all_8xx_Jun_29.json
   uv run python jasmine_data/minecraft/openai/download_videos.py --index_file_path data/open_ai_index_files/all_9xx_Jun_29.json
   uv run python jasmine_data/minecraft/openai/download_videos.py --index_file_path data/open_ai_index_files/all_10xx_Jun_29.json
   ```

3. **Preprocess videos into ArrayRecords:**
   For efficient distributed training, convert the raw videos into the arrayrecord format (make sure to have [ffmpeg](https://github.com/FFmpeg/FFmpeg) installed on your machine):

   ```bash
   cd data/
   uv run python jasmine_data/video_to_array_records.py
   ```

> **Note:** This is a large dataset and may take considerable time and storage to download and process.


<h2 name="train" id="train">Quick Start 🚀 </h2>

For a quickstart you can train the tokenizer, latent action model and dynamics model sequentially as shown below. However, note that Jasmine supports various features, which can be configured through command line arguments. Therefore, please refer to the respective training scripts. 

To train the video tokenizer, run:

```bash
uv run python jasmine/baselines/maskgit/train_tokenizer_vqvae.py \
    --ckpt_dir <path> \
    --data_dir <path>
```

To train the latent action model, run:

```bash
uv run python jasmine/baselines/train_lam.py \
    --ckpt_dir <path> \
    --data_dir <path>
```

Once the tokenizer and LAM are trained, the dynamics model can be trained with:

```bash
uv run python jasmine/train_dynamics.py \
    --tokenizer_checkpoint <path> \
    --lam_checkpoint <path> \
    --ckpt_dir <path> \
    --data_dir <path>
```

Logging with `wandb` is supported. To enable logging, set the `WANDB_API_KEY` environment variable or run:

```bash
wandb login
```

Training can then be logged by setting the `--log` flag:

```bash
uv run python jasmine/baselines/maskgit/train_tokenizer_vqvae.py \
    --ckpt_dir <path> \
    --data_dir <path> \
    --log \
    --entity <wandb-entity> \
    --project <wandb-project>
```

<h2 name="cite" id="cite">Citing 📜 </h2>

Jasmine was built by [Mihir Mahajan](https://maharajamihir.github.io/), [Alfred Nguyen](https://avocadoali.github.io/) and [Franz Srambical](https://srambical.fr/), but started as a fork of [Jafar](https://github.com/flairox/jafar), built by [Matthew Jackson](https://matthewtjackson.com) and [Timon Willi](https://www.timonwilli.com).

If you use Jasmine in your work, please cite us, Jafar, and the original Genie paper as follows:

```
@article{
    mahajan2025jasmine,
    title={Jasmine: A simple, performant and scalable JAX-based world modeling codebase},
    author={Mihir Mahajan and Alfred Nguyen and Franz Srambical and Stefan Bauer},
    journal = {p(doom) blog},
    year={2025},
    url={https://pdoom.org/jasmine.html},
    note = {https://pdoom.org/blog.html}
}
```
```
@inproceedings{
    willi2024jafar,
    title={Jafar: An Open-Source Genie Reimplemention in Jax},
    author={Timon Willi and Matthew Thomas Jackson and Jakob Nicolaus Foerster},
    booktitle={First Workshop on Controllable Video Generation @ ICML 2024},
    year={2024},
    url={https://openreview.net/forum?id=ZZGaQHs9Jb}
}
```
```
@inproceedings{
    bruce2024genie,
    title={Genie: Generative Interactive Environments},
    author={Jake Bruce and Michael D Dennis and Ashley Edwards and Jack Parker-Holder and Yuge Shi and Edward Hughes and Matthew Lai and Aditi Mavalankar and Richie Steigerwald and Chris Apps and Yusuf Aytar and Sarah Maria Elisabeth Bechtle and Feryal Behbahani and Stephanie C.Y. Chan and Nicolas Heess and Lucy Gonzalez and Simon Osindero and Sherjil Ozair and Scott Reed and Jingwei Zhang and Konrad Zolna and Jeff Clune and Nando de Freitas and Satinder Singh and Tim Rockt{\"a}schel},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=bJbSbJskOS}
}
```
