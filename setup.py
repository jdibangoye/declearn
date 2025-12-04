# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from setuptools import setup, find_packages

setup(
    name="declearn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
        "matplotlib",
        "torch",
    ],
    extras_require={
        "baselines": ["pettingzoo", "gymnasium"],
        "deep_discrete": ["pettingzoo"],
    },
)