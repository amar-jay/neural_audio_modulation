from setuptools import setup, find_packages

setup(
    name="neural_audio_compression",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project focused on neural audio compression using modulation techniques.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "librosa",
        "PyYAML",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
