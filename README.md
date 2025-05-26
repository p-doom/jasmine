<h1 align="center">Jafar: A JAX-based Genie Implementation 🧞</h1>

<p align="center">
    <a href= "https://github.com/FLAIROx/jafar/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

Jafar is a JAX-based implementation of the DeepMind paper "[Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)" (Bruce et al., 2024).

Jafar supports training of all Genie components and can complete the CoinRun reproducibility experiment (Appendix F) on a single L40S GPU in under a week.

<h2 name="start" id="start">Setup 🧗 </h2>

Jafar was built with `python 3.10` and `jax 0.4.30`. To install requirements, run:

```bash
pip install -r requirements.txt
```

Before training the models, generate the CoinRun dataset by running:

```bash
python generate_dataset.py --num_episodes 10000 --env_name coinrun
```

See [here](https://github.com/openai/procgen?tab=readme-ov-file#environments) for more environments.

Note: this is a large dataset (around 100GB) and may take a while to generate.

To generate other datasets from the gym environment run:

```bash
python generate_dataset_gym.py --num_episodes 10000 --env_name Acrobot-v1
```

See [here](https://gym.openai.com/envs/#classic_control) for more environments.

Note: This project uses gym==0.25.2 for backwards compatibility. Newer versions of gym are not supported.

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
