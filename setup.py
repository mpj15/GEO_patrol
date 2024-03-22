from setuptools import setup, find_packages

def get_version():
    path = "src/orbit_defender2d/version.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("VERSION"):
            return line.strip().split("=")[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

setup(name="geo_patrol", 
      version=get_version(),
      packages=find_packages('src'),
      package_dir={'': 'src'},
      python_requires="==3.8.19",
      install_requires=[
        "numpy==1.21.6",
        "gym==0.21.0", #really should replace gym 0.21.0 or 0.22.0
        "torch==1.10.1",
        "pettingzoo==1.15.0",
        "networkx",
        "matplotlib",
        "pyzmq",
        "tornado==6.1",
        "pygame==2.0.3",
        "bidict",
        "protobuf==3.20.0"
      ]
      )