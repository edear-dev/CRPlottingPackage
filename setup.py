from setuptools import setup, find_packages

setup(
    name="CRAB_package",             # Name of the package
    version="0.1",                   # Version number
    packages=find_packages(),        # Automatically finds 'CRAB_package' folder
    
    # List any other libraries your package needs to run
    # (Example: if you use numpy or pandas, list them here)
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'typing'
    ],
)