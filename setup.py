from setuptools import find_packages, setup

setup(
    name="oh-comp-kernels",
    version="1.0.0",
    author="Giorgio Tosti Balducci",
    author_email="giotb92@gmail.com",
    description="Classify ultimate failure of open-hole composite laminates with classical and quantum kernel SVMs",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "jax",
        "jaxlib",
        "jaxopt",
        "optax",
        "pennylane",
        "scikit-learn",
        "torch",
        "zenodo-get",
    ],
    extras_require={
        "plotting": ["matplotlib", "seaborn"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
