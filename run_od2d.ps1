# Windows Powershell script to open a new command window, cd to ~\GEO_patrol, run conda activate geo_patrol, and then run python od2d.py
# This script is used to run the od2d.py script in a new command window with the correct environment activated
#
# Copyright (c) 2024, Michael P. Jones (mpj@alum.mit.edu)
# SPDX-License-Identifier: MIT


$directory = "."
$conda_env = "geo_patrol"
$script = "run_geo_patrol.py"

# Open a new command window

Start-Process cmd -ArgumentList "/k cd $directory && conda activate $conda_env && python $script" -NoNewWindow

# End of script


# Optional:
# Create a desktop shortcut that points to this file.
# Then right click properties - change icon and change the icon the od2d_icon.ico file in this folder.

# Adjust target to be: "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File <full_path>\run_od2d.ps1"
# Adjust start in to be: "<full_path>\GEO_patrol"
# Replace <full_path> with the full path to the folder where this script is located.
