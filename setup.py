import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

NAME = "vs-debandshit"
# VERSION = "0.4.2"
RELEASE = "0.4.2"

setuptools.setup(
    name=NAME,
    version=RELEASE,
    author="LightArrowsEXE",
    author_email="Lightarrowsreboot@gmail.com",
    description="VapourSynth Debanding Functions Collection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["debandshit"],
    url="https://github.com/LightArrowsEXE/debandshit",
    package_data={
        'debandshit': ['py.typed'],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
