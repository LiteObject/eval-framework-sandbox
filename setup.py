from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="eval-framework-sandbox",
    version="0.1.0",
    description="Sandbox project for experimenting with QA evaluation frameworks",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "eval": [
            "pytest>=7.4.0",
            "langchain>=0.0.350",
            "langchain-community>=0.0.28",
            "ragas>=0.0.15",
            "openai>=1.0.0",
            "deepeval>=0.20.0",
        ]
    },
    python_requires=">=3.10",
)
