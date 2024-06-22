### GEO Patol
Welcome to GEO Patrol. This is a two-player, perfect information, simultaneous move, discrete game based on the Orbit Defender 2D (OD2D) environment. OD2D was originally developed by MIT Lincoln Laboratory as an environment for modelling non-cooperative satellite interactions at a high-level of abstraction.

For more information, see the rulebook: https://github.com/mpj15/GEO_patrol/blob/main/GEO_Patrol_Rule_Book.pdf

Also the OD2D repo's wiki: https://github.com/mpj15/spacegym-od2d/wiki

GEO Patrol is an asymmetric version of the baseline OD2D game. In GEO Patrol, one side plays offense, with more Patrol satellites with more fuel but no ammo. The other side plays defense, with fewer Patrol satellites and less fuel but more firepower.

Developed by Michael Jones, mpj@alum.mit.edu.

Distributed under an MIT License. See license file and SPDX file for details.

## Installation

To install, run the following in a terminal with conda installed:

```bash
conda env create -f environment.yml
conda activate geo_patrol
```
Otherwise, you can install the dependencies with pip:

```bash
pip install -r requirements.txt
```

## Running the Game
After installing and activating the environment, you can run the game with the following command:

```bash
python run_geo_patrol.py
```

Or, in Widows, you can run the powhershell script:

```bash 
.\run_game.ps1
```

Finally, if you'd like, you can make a shortcut to the powershell script, and set the shortcut icon to the ico for a click to open game window. Follow the instructions in the .ps1 file's comments for more details.
