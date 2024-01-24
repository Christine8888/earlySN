from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="earlySN",
    version="0.0.1",
    author="Christine Ye",
    author_email="cye@stanford.edu",
    description="A package to fit lightcurves with early excesses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Christine8888/earlySN",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
