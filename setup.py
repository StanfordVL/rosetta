from setuptools import setup, find_packages

setup(
    name='rosetta',
    version='0.1.0',
    author="Sanjana Srivastava, Kangrui Wang, Yung-Chieh Chan",
    author_email="sanjana2@stanford.edu",
    packages=find_packages(),
    install_requires=[
        "astor",
        "ipython",
        "openai",
        "opencv_python",
        "Pillow",
        "redbaron",
        "setuptools",
        "fire"
    ]
)