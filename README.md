<h1 align="center">Jafar: A JAX-based Genie Implementation 🧞</h1>

<p align="center">
    <a href= "https://github.com/FLAIROx/jafar/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

This is a feature-complete fork of Jafar, a JAX-based implementation of the DeepMind paper "[Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)" (Bruce et al., 2024).

Jafar supports training of all Genie components and can complete the CoinRun reproducibility experiment (Appendix F) on a single L40S GPU in under a week.

This repository implements bugfixes and multitudes of additional features described below.

<h2 name="overview" id="overview">Overview</h2>

- Distributed checkpointing & loading
    - also onto a different hardware topology (e.g. initial training on 1 node --> reinitialization onto 4 nodes)
- Asynchronous checkpointing
- Optimized dataloading (each GPU loads its shard in parallel)
- Model+optimizer states+dataloader checkpointing (just resume your training run and everything is taken care of)
- Full reproducibility with exact training curves (seeded dataloading + training)
- Automatic checkpoint management (automatic deletion/retention according to specified retention policy)
- Mixed precision (with correct full precision casting, e.g. before softmax)
- Flash attention (from cudnn, which is more optimized than Tri Dao's)
- int8-quantization is on the roadmap (via https://github.com/google/aqt)
- KV caching for inference of causal Transformer (still in PR)
    - Currently working on frame-level KV cache resets for accelerated spatiotemporal attention
- Activation checkpointing (even onto host memory if desired)
- Distributed Data parallelism (changing to FSDP requires changing one line of code)
- wandb, cli and tensorboard logging (tb logging still in PR)
- Just-in-time compiled train step
- Cosine/ WSD learning rate schedules (no need to retrain if you realize that you want to train for longer)
- Index shuffling during dataloading (doesn't require loading the dataset into memory for shuffling)
- 'google-native' stack
    - https://github.com/google/orbax for checkpointing
    - https://github.com/google/grain for dataloading
    - https://github.com/google-deepmind/dm_pix for image manipulation
    - https://github.com/google/array_record as the data format
- essentially, no other dependencies
- We are currently working on migrating to the new flax.nnx API, after which we will also have:
    - significantly less boilerplate
    - easy model surgery
    - model inspection

<h2 name="start" id="start">Setup 🧗 </h2>

Jafar was built with `python 3.10` and `jax 0.4.30`. To install requirements, run:

```bash
pip install -r requirements.txt
pre-commit install
```

Before training the models, generate the CoinRun dataset by running:

```bash
python generate_dataset.py --num_episodes 10000
```

Note: this is a large dataset (around 100GB) and may take a while to generate.

For performant distributed training, we additionally preprocess the dataset into `TFRecord`s:

```bash
python preprocess_dataset.py
```

<h2 name="train" id="train">Quick Start 🚀 </h2>

Genie has three components: a [video tokenizer](models/tokenizer.py), a [latent action model](models/lam.py), and a [dynamics model](models/dynamics.py). Each of these components are trained separately, however, the dynamics model requires a pre-trained video tokenizer and latent action model.

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

<h2 name="cite" id="cite">Citing Jafar 📜 </h2>

Jafar was built by [Matthew Jackson](https://matthewtjackson.com) and [Timon Willi](https://www.timonwilli.com).

If you use Jafar in your work, please cite us and the original Genie paper as follows:

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
