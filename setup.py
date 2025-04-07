from setuptools import setup, find_packages

setup(
    name="localflow-ml",
    version="0.1.0",
    description="LocalFlow-ML: A Local MLOps Pipeline",
    author="Debarghya Saha",
    author_email="info@example.com",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "mlflow>=2.7.1",
        "evidently>=0.4.5",
        "fastapi>=0.103.1",
        "uvicorn>=0.23.2",
        "boto3>=1.28.40",
        "prometheus-client>=0.17.1",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train-model=pipelines.training.train:main",
            "detect-drift=monitoring.evidently.drift_detection:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 