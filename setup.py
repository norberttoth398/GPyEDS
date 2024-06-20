import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GPyEDS",
    version="0.0.1",
    author="norberttoth398",
    author_email="nt398@cam.ac.uk",
    description="Autoencoder-based unsupervised feature extraction toolbox.",
    long_description=long_description,
    url="https://github.com/norberttoth398/GPyEDS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires = install_requires,
)
