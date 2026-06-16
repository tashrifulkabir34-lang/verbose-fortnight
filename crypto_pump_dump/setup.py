from setuptools import setup, find_packages
setup(
    name="crypto-pump-dump-replication",
    version="1.0.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.10",
)
