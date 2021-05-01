# DCGAN

## Setup

```shell
> poetry install
> poetry run poe torch-cuda11
```

## Download training images

```shell
> poetry run python src/get_data.py
```

## Training

```shell
> poetry run python src/train.py
```

## Generate images

```shell
> poetry run python src/predict.py
```