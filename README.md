# Hackathon starter-kit

Zero-shot Diffusion based image restoration.

The goal of the hackathon is to guide the participants to develop robust and efficient algorithms
that leverage the advances in DMs to solve inverse problems with no-additional training of
the models.
The focus will be common tasks encountered in image restoration such as image inpainting and Super Resolution.


## Installation

Beforehand, ensure to download the code.
You can use ``git`` or download it as ``zip``.

1. Run the following the command to create a fresh Python environment.

```bash
python3 -m venv venv-hackathon
```

1. Activate the environment

```bash
source venv-hackathon/bin/activate
```

1. then install the project on editable mode

```bash
pip install -e .
```

1. Finally, download [FFHQ model checkpoint](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh) and put it on ``material/checkpoints`` folder.

Do not forget to put the absolute path of the project in ``py_source/local_paths.py``.
Similarly, put the absolute path to FFHQ checkpoint in ``/py_source/configs/ffhq_model.yaml``


## About the repository structure

The ``material`` folder contains external files such images, and model checkpoints.

Essential functions and classes to load pre-trained Diffusion Models, load images, display them, and initialize inverse problem are located in ``py_source/`` folder.
In particular,
- ``py_source/sampling/`` folder contains examples of algorithm for solving inverse problem
- ``py_source/utils.py`` contains functions to load model, images, and plot them

There are two notebooks to help you get started
- ``demo_inverse_problems.ipynb`` shows how to define an inverse problem, solve it with an algorithm, and visualize the result
- ``demo_evaluation.ipynb`` explains and illustrates the evaluation process of an algorithm


## Note

- Link to download FFHQ model checkpoint https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh
- Evaluation script and the inverse problems used will be uploaded later during the week
