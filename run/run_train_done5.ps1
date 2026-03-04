# Save current directory (folder1)
$cwd = Get-Location

# Set PYTHONPATH to your project root
$env:PYTHONPATH = "C:\Users\abzza\PycharmProjects\dif_driven_maddpg"

# Activate the virtual environment
& "C:\Users\abzza\PycharmProjects\dif_driven_maddpg\.venv3.10\Scripts\Activate.ps1"

# Run the script (but keep working directory as $cwd so relative paths work)
python "C:\Users\abzza\PycharmProjects\dif_driven_maddpg\run\train_done5.py"
