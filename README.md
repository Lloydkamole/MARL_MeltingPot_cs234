# Melting Pot

*A suite of test scenarios for multi-agent reinforcement learning.*

<!-- linter off -->
<!-- GITHUB -->
[![Python](https://img.shields.io/pypi/pyversions/dm-meltingpot.svg)](https://pypi.python.org/pypi/dm-meltingpot)
[![PyPI version](https://img.shields.io/pypi/v/dm-meltingpot.svg)](https://pypi.python.org/pypi/dm-meltingpot)
[![PyPI tests](../../actions/workflows/pypi-test.yml/badge.svg)](../../actions/workflows/pypi-test.yml)
[![Tests](../../actions/workflows/test-meltingpot.yml/badge.svg)](../../actions/workflows/test-meltingpot.yml)
[![Examples](../../actions/workflows/test-examples.yml/badge.svg)](../../actions/workflows/test-examples.yml)
<!-- /GITHUB -->
<!-- linter on -->

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(IMAGE_ALT_TEXT_INACCESSIBLE) -->

<div align="center">
  <img src="https://github.com/google-deepmind/meltingpot/blob/main/docs/images/meltingpot_montage.gif?raw=true"
       alt="Melting Pot substrates"
       height="250" width="250" />
</div>

[Melting Pot 2.0 Tech Report](https://arxiv.org/abs/2211.13746)
[Melting Pot Contest at NeurIPS 2023](https://www.aicrowd.com/challenges/meltingpot-challenge-2023)

## 🧠 CS234 Project Contribution: MARL in Melting Pot

This repository contains our team's project contributions for Stanford's CS234 (Reinforcement Learning) class. Our research focuses on exploring multi-agent reinforcement learning (MARL) dynamics within the **Google DeepMind Melting Pot** environment.

### 📍 Where to Find Our Work

Our primary codebase, custom environment wrappers, and experimental setups are housed entirely within the [`examples/rllib`](./examples/rllib) directory.

### 🛠️ Core Implementations

Inside this directory, our specific technical contributions include:
* **Environment Wrapper (`utils.py`):** We built `MeltingPotEnv`, a custom adapter that bridges DeepMind's `dmlab2d` environment with Ray RLlib, converting observation and action tuples into the `spaces.Dict` format required for multi-agent training.
* **PPO Training Pipeline (`self_play_train.py`):** We implemented a self-play training script utilizing Proximal Policy Optimization (PPO). The model architecture features a custom Convolutional Neural Network (CNN) specifically tuned for Melting Pot's 8x8 visual sprites, paired with an LSTM (cell size 256) to handle partial observability and memory.
* **Evaluation & Rendering (`view_models.py`):** A custom Pygame-based evaluation script that automatically fetches the best-performing PPO checkpoint and renders the multi-agent rollout at 5 frames per second for visual analysis.

### 🚀 Getting Started

To run our CS234 implementations, navigate to our contribution directory and execute the scripts:

```bash
# Navigate to our specific project directory
cd examples/rllib

# Launch the PPO self-play training pipeline
python self_play_train.py

# Visualize the best trained agents
python view_models.py


## Installation

Melting Pot is available on PyPI](https://pypi.python.org/pypi/dm-meltingpot)
and can be installed using:

```shell
pip install dm-meltingpot
```

After doing this you can then `import meltingpot` in your own code.

NOTE: Melting Pot is built on top of [DeepMind Lab2D](https://github.com/google-deepmind/lab2d)
which is distributed as pre-built wheels. If there is no appropriate wheel for
`dmlab2d`, you will need to build it from source (see
[the `dmlab2d` `README.md`](https://github.com/google-deepmind/lab2d/blob/main/README.md)
for details).

## Development

### Codespace

The easiest way to work on the Melting Pot source code, is to use our
pre-configured development environment via a
[Github CodeSpace](https://github.com/features/codespaces).

This provides a tested development workflow that allows for reproducible builds,
and minimizes dependency management. We strongly advise preparing all Pull
Requests for Melting Pot via this workflow.

### Manual setup

If you want to work on the Melting Pot source code within your own development
environment you will have to handle installation and dependency management
yourself.

For example, you can perform an editable installation as follows:

1.  Clone Melting Pot:

    ```shell
    git clone -b main https://github.com/google-deepmind/meltingpot
    cd meltingpot
    ```

2.  Create and activate a virtual environment:

    ```shell
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install Melting Pot:

    ```shell
    pip install --editable .[dev]
    ```

4.  Test the installation:

    ```shell
    pytest --pyargs meltingpot
    ```

## Example usage

### Evaluation
The [evaluation](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/utils/evaluation/evaluation.py) library can be used
to evaluate [SavedModel](https://www.tensorflow.org/guide/saved_model)s
trained on Melting Pot substrates.

Evaluation results from the [Melting Pot 2.0 Tech Report](https://arxiv.org/abs/2211.13746)
can be viewed in the [Evaluation Notebook](https://github.com/google-deepmind/meltingpot/blob/main/notebooks/evaluation_results.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/meltingpot/blob/main/notebooks/evaluation_results.ipynb)

### Interacting with the substrates

You can try out the substrates interactively with the
[human_players](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/human_players) scripts. For example, to play
the `clean_up` substrate, you can run:

```shell
python meltingpot/human_players/play_clean_up.py
```

You can move around with the `W`, `A`, `S`, `D` keys, Turn with `Q`, and `E`,
fire the zapper with `1`, and fire the cleaning beam with `2`. You can switch
between players with `TAB`. There are other substrates available in the
[human_players](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/human_players) directory. Some have multiple
variants, which you select with the `--level_name` flag.

### Training agents

We provide an illustrative example script using
[RLlib](https://github.com/ray-project/ray). However, note
that Melting Pot is agnostic to how you train your agents, and this
script is not meant to be a suggestion for how to achieve a good score
in the task suite. The authors of the suite never used this example training
script in their own work.

#### RLlib

This example uses RLlib to train agents in
self-play on a Melting Pot substrate.

First you will need to install the dependencies needed by the examples:

```shell
cd <meltingpot_root>
pip install -r examples/requirements.txt
```

Then you can run the training experiment using:

```shell
cd examples/rllib
python self_play_train.py
```

## Documentation

Full documentation is available [here](https://github.com/google-deepmind/meltingpot/blob/main/docs/index.md).

## Citing Melting Pot

If you use Melting Pot in your work, please cite the accompanying articles:

```bibtex
@inproceedings{leibo2021meltingpot,
    title={Scalable Evaluation of Multi-Agent Reinforcement Learning with
           Melting Pot},
    author={Joel Z. Leibo AND Edgar Du\'e\~nez-Guzm\'an AND Alexander Sasha
            Vezhnevets AND John P. Agapiou AND Peter Sunehag AND Raphael Koster
            AND Jayd Matyas AND Charles Beattie AND Igor Mordatch AND Thore
            Graepel},
    year={2021},
    journal={International conference on machine learning},
    organization={PMLR},
    url={https://doi.org/10.48550/arXiv.2107.06857},
    doi={10.48550/arXiv.2107.06857}
}
```

```bibtex
@article{agapiou2022melting,
  title={Melting Pot 2.0},
  author={Agapiou, John P and Vezhnevets, Alexander Sasha and Du{\'e}{\~n}ez-Guzm{\'a}n, Edgar A and Matyas, Jayd and Mao, Yiran and Sunehag, Peter and K{\"o}ster, Raphael and Madhushani, Udari and Kopparapu, Kavya and Comanescu, Ramona and Strouse, {DJ} and Johanson, Michael B and Singh, Sukhdeep and Haas, Julia and Mordatch, Igor and Mobbs, Dean and Leibo, Joel Z},
  journal={arXiv preprint arXiv:2211.13746},
  year={2022}
}
```

## Disclaimer

This is not an officially supported Google product.
