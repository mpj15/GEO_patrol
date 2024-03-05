# Write a powershell script to open a new command window, cd to ~\GEO_patrol, run conda activate geo_patrol, and then run python od2d.py
# This script is used to run the od2d.py script in a new command window with the correct environment activated

$directory = "."
$conda_env = "geo_patrol"
# $script = "checkpoint_just_torch_CLI.py"
$script = "od2d.py"

# Open a new command window

Start-Process cmd -ArgumentList "/k cd $directory && conda activate $conda_env && python $script" -NoNewWindow

# End of script


#On windows create a desktop shortcut that points to this file.
# Then right click properties - change icon and change the icon the
# od2d_icon.ico file in this folder.

# Adjust target to be: C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File ~full_path_to~\run_od2d.ps1
# Adjust start in to be: ~\GEO_patrol
