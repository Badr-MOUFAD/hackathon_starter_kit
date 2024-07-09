from setuptools import setup

requirements =  [
    "torch",
    "torchvision",
    "transformers",
    "diffusers[torch]",
    "lpips",
    "Pillow",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "omegaconf",
    "tqdm",
    "PyYAML",
]

setup(
    name='py_source',
    version='0.0.0',
    description="Starter code for the Hackathon",
    install_requires=requirements,
)
