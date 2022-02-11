from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('ssvm_evaluation/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="ssvm_evaluation",  # TODO: This name is already in PyPi. We need to choose something different.
    version=main_ns["__version__"],
    license="MIT",
    packages=find_packages(exclude=["tests", "examples", "*.ipynb"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "scipy",
        "more-itertools",
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="Experimental results and analysis of our SSVM MS and RT score integration.",
    url="https://github.com/aalto-ics-kepaco/msms_rt_ssvm_experiments",
)
