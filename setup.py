from setuptools import find_packages, setup

install_requires = []


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


if __name__ == "__main__":
    setup(
        name="dacon_gps",
        version="1.0.0",
        description="Dacon GPS multitask model inference",
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="kerobro",
        keywords="multitask model, Graph Transformer",
        packages=find_packages(
            exclude=(
                "saved_model",
                "dataset",
            )
        ),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.8.0",
        install_requires=install_requires,
        ext_modules=[],
        zip_safe=False,
    )
