import sys
import os
from cx_Freeze import setup, Executable

#Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"include_files":["exiftool.exe",
                                      r"C:\Program Files\Python36\DLLs\tcl86t.dll",
                                      r"C:\Program Files\Python36\DLLs\tk86t.dll"],
                     "namespace_packages":["scipy"],
                     "packages":["os", "numpy", "scipy", "cv2", "gdal",
                                 "osgeo.gdal_array", "tkinter", "matplotlib"],
                     "excludes":["WXAgg", "PyQt5", "gtk"]}

#Set TCL and TK Library manually
os.environ["TCL_LIBRARY"] = r"C:\Program Files\Python36\tcl\tcl8.6"
os.environ["TK_LIBRARY"] = r"C:\Program Files\Python36\tcl\tk8.6"

#GUI applications require a different base on Windows (the default is for a console appliction)
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(name = "Empirical_Correction_App",
      version = "0.1",
      description = "Empirical line correction Application!",
      options = {"build_exe":build_exe_options},
      executables = [Executable("Rad_Emp_correction.pyw", base=base)])
