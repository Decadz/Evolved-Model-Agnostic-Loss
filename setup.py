from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "pandas",
    "deap",
    "torch",
    "higher",
    "torchvision",
    "pyyaml",
    "matplotlib",
    "scikit-learn",
    "tqdm"
]

setup(
    name="LossFunctionLearning",
    author="Christian Raymond",
    author_email="christianfraymond@gmail.com",
    version=1.0,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
