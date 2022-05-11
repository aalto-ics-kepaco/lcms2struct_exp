from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('ssvm_evaluation/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="ssvm_evaluation",
    version=main_ns["__version__"],
    license="MIT",
    packages=find_packages(exclude=["tests", "examples", "*.ipynb"]),

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="Tools to produce the figures in our LC-MS2Struct manuscript.",
    url="https://github.com/aalto-ics-kepaco/lcms2struct_exp",
)
