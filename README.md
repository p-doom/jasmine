<h1 align="center">🧞‍♀️ Jasmine: A simple, performant and scalable JAX-based world modeling codebase 🧞‍♀️</h1>

<p align="center">
    <a href= "https://github.com/FLAIROx/jafar/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

Jasmine is a production-ready JAX-based world modeling codebase. It currently implements the high-level architecture of [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) (Bruce et al., 2024) with [MaskGIT](https://arxiv.org/abs/2202.04200) (Chang et al., 2022), as well as an autoregressive (causal) baseline. A diffusion baseline is coming soon.

Jasmine scales from single hosts to hundreds of xPUs thanks to XLA and strives to be an easily hackable, batteries-included foundation for world modeling research.

<h2 name="overview" id="overview">Overview</h2>

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
- Easy model surgery thanks to the new [flax.nnx](https://flax.readthedocs.io/en/latest/migrating/linen_to_nnx.html) API
- [Shape suffixes](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd) throughout the repository

<h2 name="start" id="start">Setup 🧗 </h2>

Jasmine requires `python 3.10`, `jax 0.6.2` and `flax 0.10.7`. To install the requirements, run:

```bash
pip install -r requirements.txt
pre-commit install
```

Download OpenAI's VPT dataset by running:

```bash
bash input_pipeline/download/openai/download_index_files.sh
python input_pipeline/download/openai/download_videos.py
```

Note: this is a large dataset and may take a while to download.

For performant distributed training, we additionally preprocess the dataset into `arrayrecords`:

```bash
python input_pipeline/preprocess/video_to_array_records.py
```

<h2 name="train" id="train">Quick Start 🚀 </h2>

Genie has three components: a [video tokenizer](models/tokenizer.py), a [latent action model](models/lam.py), and a [dynamics model](models/dynamics.py). Each of these components are trained separately, however, the dynamics model requires a pre-trained video tokenizer (and latent action model).

To train the video tokenizer (similar for the LAM), run:

```bash
python train_tokenizer.py --ckpt_dir <path>
```

Once the tokenizer and LAM are trained, the dynamics model can be trained with:

```bash
python train_dynamics.py --tokenizer_checkpoint <path> --lam_checkpoint <path>
```

Logging with `wandb` is supported. To enable logging, set the `WANDB_API_KEY` environment variable or run:

```bash
wandb login
```

Training can then be logged by setting the `--log` flag:

```bash
python train_tokenizer.py --log --entity <wandb-entity> --project <wandb-project>
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
    url={https://pdoom.org/jasmine.html}
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
