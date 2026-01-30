from setuptools import setup, find_packages

setup(
    name="Topsis-Reet-102303532",
    version="0.0.1",
    author="Reet",
    description="TOPSIS command line tool",
    packages=find_packages(),
    install_requires=["pandas", "numpy"],
    entry_points={
        'console_scripts': [
            'topsis=topsis.topsis:main'
        ]
    },
)
