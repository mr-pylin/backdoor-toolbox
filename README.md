# 🚪 Backdoor Toolbox

[![License](https://img.shields.io/github/license/mr-pylin/backdoor-toolbox?color=blue)](https://github.com/mr-pylin/backdoor-toolbox/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.12.8-yellow?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3128/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0ecd3ef1082844d8b0c1227bedc2a59d)](https://app.codacy.com/gh/mr-pylin/backdoor-toolbox/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
![Repo Size](https://img.shields.io/github/repo-size/mr-pylin/backdoor-toolbox?color=lightblue)
![Last Updated](https://img.shields.io/github/last-commit/mr-pylin/backdoor-toolbox?color=orange)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?color=brightgreen)](https://github.com/mr-pylin/backdoor-toolbox/pulls)

A modular toolbox for researching and benchmarking backdoor attacks and defenses in deep neural networks.

## Overview

`backdoor-toolbox` is a lightweight, modular framework designed to accelerate research on **backdoor attacks and defenses** in deep neural networks. This toolbox provides a unified interface for injecting, training on, evaluating, and defending against poisoned datasets across standard benchmarks such as MNIST and CIFAR-10.

The core philosophy of this repository is to be:

- 🧩 **Modular**: Attacks, defenses, models, datasets, and evaluation metrics are decoupled and easily swappable.
- 🧪 **Research-friendly**: Every component is designed with minimal abstraction for quick debugging, experimentation, and extension.
- ⚙️ **Configurable**: A simple `config.py` setup allows easy control over experiments.
- 🪶 **Lightweight**: No heavy dependencies or bloated abstractions—only essential code to make your experiments reproducible and transparent.

This toolbox is ideal for:

- Reproducing known backdoor attack/defense methods.
- Benchmarking your own method against existing ones.
- Understanding how subtle changes affect attack success rate (ASR), clean data accuracy (CDA) and robust-accuracy.

## Repository Structure

The project is organized for clarity, extensibility, and research workflows. Below is a high-level overview of the main directories and scripts:

```
.
├── src/backdoor_toolbox/    # Core package: attacks, defenses, datasets, models, routines
│   ├── datasets/            # Loading logic per dataset
│   ├── models/              # Model architectures (e.g., ResNet variants)
│   ├── routines/            # Training/evaluation routines (e.g., BaseRoutine, MultiAttackRoutine)
│   │     ├── attacks/       # Attack strategies (e.g., multi-attack with various triggers)
│   │     ├── defenses/      # Defense strategies (e.g., Ensemble, Model-Aggregation)
│   │     └── neutral/       # Training a clean model as baseline
│   ├── triggers/            # Trigger patterns and transformations added before feeding the model
│   └── utils/               # Logger, metrics, helpers
├── assets/                  # Visual outputs, figures, and Grad-CAM images
├── poetry.lock              # Locked dependencies for reproducibility
├── pyproject.toml           # Poetry configuration file
└── README.md
```

## ⚙️ Setup

This section walks you through installing dependencies and setting up your first experiment.  
The project requires Python **v3.10** or higher. It was developed and tested using Python **v3.12.8**. If you encounter issues running the specified version of dependencies, consider using this version of Python.

### 📝 List of Dependencies

[![ipykernel](https://img.shields.io/badge/ipykernel-6.29.5-ff69b4)](https://pypi.org/project/ipykernel/6.29.5/)
[![ipywidgets](https://img.shields.io/badge/ipywidgets-8.1.5-ff6347)](https://pypi.org/project/ipywidgets/8.1.5/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-green)](https://pypi.org/project/matplotlib/3.10.0/)
[![numpy](https://img.shields.io/badge/numpy-2.2.1-orange)](https://pypi.org/project/numpy/2.2.1/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-darkblue)](https://pypi.org/project/scikit-learn/1.6.0/)
[![torch](https://img.shields.io/badge/torch-2.5.1%2Bcu124-gold)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-2.5.1%2Bcu124-lightgreen)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.20.1%2Bcu124-teal)](https://pytorch.org/)
[![torchinfo](https://img.shields.io/badge/torchinfo-1.8.0-blueviolet)](https://pypi.org/project/torchinfo/1.8.0/)
[![torchmetrics](https://img.shields.io/badge/torchmetrics-1.6.1-lightgray)](https://pypi.org/project/torchmetrics/1.6.1/)

### 📦 Install Dependencies

Use [**Poetry**](https://python-poetry.org/) for dependency management. It handles dependencies, virtual environments, and version locking more efficiently than pip.
To install all dependencies **and** the current project as a package:

```bash
poetry install
```

## 🛠️ Usage Instructions

Before running any experiment, you must **configure two files** to specify what to run and how to run it:

### 1. Global Routine Configuration

Open `src/backdoor_toolbox/config.py` and **uncomment exactly one** `routine = {...}` block corresponding to the experiment type you want to execute.

Each block specifies:

- The root module path (e.g., `routines.attacks.multi_attack`)
- The file name (e.g., `multi_attack.py`)
- The class name (e.g., `MultiAttackRoutine`)
- Whether to enable verbose mode

Example:

```python
# [ATTACK]
routine = {
    "root": "backdoor_toolbox.routines.attacks.multi_attack",
    "file": "multi_attack",
    "class": "MultiAttackRoutine",
    "verbose": True,
}
```

### 2. Routine-Specific Configuration

Each routine directory (e.g., `routines/attacks/multi_attack/`) contains its own `config.py` where you define:

- Dataset and preprocessing
- Model architecture
- Trigger types and poison parameters
- Training/testing hyperparameters
- Logger and output settings
- Optional features like Grad-CAM, feature maps, or prediction demos

> 🔍 These configs give you full control over the experiment pipeline, dataset splits, triggers, backdoor types, model saving, evaluation metrics, etc.

### ⚠️ Dataset-Specific Notes

- For **CIFAR-10**, the default trigger blending strength `alpha` used for MNIST is too weak to be effective.  
  You must manually modify the file:

  `triggers/transform/transform.py\

  and update the line:

  ```python
  alpha = random.uniform(0.05, 0.1)  # for MNIST
  ```

  to:

  ```python
  alpha = random.uniform(0.15, 0.2)  # for CIFAR-10
  ```

> 💡 This change reflects stronger trigger visibility required for natural image datasets like CIFAR-10.

### ✅ Running the Pipeline

Once both configuration files are set, execute:

```bash
poetry run python src/backdoor_toolbox/main.py
```

This structure makes it easy to plug in new datasets, models, or trigger types.

## 🔍 Find Me

Any mistakes, suggestions, or contributions? Feel free to reach out to me at:

- 📍[**linktr.ee/mr_pylin**](https://linktr.ee/mr_pylin)

I look forward to connecting with you! 🏃‍♂️

## 📄 License

This project is licensed under the **[Apache License 2.0](./LICENSE)**.  
You are free to **use**, **modify**, and **distribute** this code, but you **must** include copies of both the [**LICENSE**](./LICENSE) and [**NOTICE**](./NOTICE) files in any distribution of your work.

<!-- ### ©️ Copyright Information

- **Assets**:
  - The images located in the [./assets/blend_trigger/](./assets/blend_trigger/) folder are licensed under the **[CC BY-ND 4.0](./assets/images/original/LICENSE)**. -->

<!-- ## 🧪 Citation & Paper

This toolbox was primarily developed as part of my academic research on backdoor attacks and defenses in deep neural networks.

If you use this repository in your own work or find it helpful, please consider citing the corresponding paper below:

> **[Your Paper Title Here]**  
> [Authors...]  
> *Conference or Journal Name, Year*  
> [📄 Paper Link (arXiv, DOI, etc.)]  
> [🔗 GitHub Repo: https://github.com/mr-pylin/backdoor-toolbox](https://github.com/mr-pylin/backdoor-toolbox)

BibTeX:

```bibtex
@article{your2025paper,
  title={Your Paper Title},
  author={Amirhossein [Last Name] and others},
  journal={Conference/Journal Name},
  year={2025},
  url={https://github.com/mr-pylin/backdoor-toolbox}
}
```

> For reproducibility, all experiments from the paper can be reproduced using this toolbox with minor configuration changes. -->
