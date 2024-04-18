### GEO Patol
Welcome to GEO Patrol. This is a two-player, perfect information, simultaneous move, discrete game, based on the Orbit Defender 2D (OD2D) environment. OD2D was originally developed by MIT Lincoln Laboratory.

For more information, see the OD2D repo here: https://github.com/mpj15/spacegym-od2d 

GEO Patrol is an asymmetric version of the baseline OD2D game. In GEO Patrol, one side plays offense, with more Patrol satellites with more fuel but no ammo. The other side plays defense, with fewer Patrol satellites and less fuel but more firepower.

A rule book is included in this repo and a more info can also be found at the OD2D Repo's Wiki, here: https://github.com/mpj15/spacegym-od2d/wiki

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
python od2d.py
```

Or, in Widows, you can run the powhershell script:

```bash 
.\run_game.ps1
```

Finally, if you'd like, you can make a shortcut to the powershell script, and set the shortcut icon to the ico for a click to open game window. Follow the instructions in the .ps1 file's comments for more details.
