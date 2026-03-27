from setuptools import setup, find_packages

setup(
    name="llm-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    description="A complete and bug-free LLM training and inference framework.",
    author="AI Assistant",
)
