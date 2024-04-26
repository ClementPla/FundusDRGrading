![header](imgs/header.png)

# Fundus DR Grading

[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye-up.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/index.html)
[![Lightning](https://img.shields.io/badge/Lightning-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)

## Description

This project aims to evaluate the performance of different models for the classification of diabetic retinopathy (DR) in fundus images. The reported perfomance metrics are not always consistent in the literature. Our goal is to provide a fair comparison between different models using the same datasets and evaluation protocol.

## Installation

To install the project and its dependencies, you can use the following commands:

```sh
pip install -r requirements.lock
pip install .
```

## Usage
To train the model, you can use the [train script](src/fundusClassif/scripts/train.py) entrypoint:

```sh
train --model <model_name>
```

## Development
This project uses pre-commit for managing and maintaining pre-commit hooks. Make sure to install the dev dependencies:
```sh
pip install -r requirements-dev.lock
pip install -e .
```

For better project management, we recommend using [Rye](https://rye-up.com/) to manage dependencies, building, and testing the project.

```sh
curl -sSf https://rye-up.com/get | bash
rye sync --all-features
```

Then install the pre-commit hooks:
```sh
pre-commit install
rye run pre-commit install # With Rye
```
