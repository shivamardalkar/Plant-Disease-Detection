import sys
import os 
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "include_files": ["images/", "evaluation.db", "GUI_Master_old.py", "login.py", "plant_model.h5", "registration.py", "CNNModel.py"],
    'packages': ['PIL', 'tensorflow']
}

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name="CNPlant",
    version="0.1",
    description="My GUI application!",
    options={"build_exe": build_exe_options},
    executables=[Executable("GUI_Master_old.py", target_name='masterexe', base=base, icon="leaf.ico")],
)