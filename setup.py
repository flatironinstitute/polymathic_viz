from setuptools import setup

url = "https://github.com/AaronWatters/array_gizmos"
version = "0.1.0"
readme = open('README.md').read()

setup(
    name="polymathic_viz",
    packages=[
        "polymathic_viz",
        ],
    version=version,
    description="Tools for creating visualizations that are useful for the polymathic project.",
    long_description=readme,
    long_description_content_type="text/markdown",
    #include_package_data=True,
    author="Aaron Watters",
    author_email="awatters@flatironinstitute.org",
    url=url,
    install_requires=[
        #"jp_doodle",
        "numpy",
        #"H5Gizmos",
        "h5py",
        "matplotlib",
        "opencv-python",
        ],
    scripts = [
    ],
    python_requires=">=3.6",
)
