# EmpCorrect_App
Use empirical line correction method to convert mosaic image to at-surface reflectance image

# This is my first GUI application so that I just take any modules for quick development.
# Any Suggestion for improvement is welcome

Note: The script is coded in Python 3.6 environment on Windows 10 64-bit, and may not work on version < 3.5 or other platform.
Dependency: numpy, scipy, opencv, gdal, matplotlib

# Usage:
Download the build folder and the Windows short cut for quick usage. Double click the short cut to execute the application.
Use mode 1 to zoom with mouse drag (right click to reset view), and mode 0 to draw polygon to specify pixels (right click to cancel).
After drawing the area of interest, input the at-surface reflectance value for those pixels.
When finish, close the "Original" window and it will generate the linear regression graph.
If you satisfy the result, close the graph and confirm to continue. The application will apply the linear equation to calculate the at-surface reflectance image.

# Warning:
The latest matplotlib version uses TkAgg as its default backend, and this code will not run successfully with this configuration.
If you want to run the script instead of the executable, change the backend to PyQt4
The matplotlib config file is located at: %PYTHONPATH%\Lib\site-packages\matplotlib\mpl-data\matplotlibrc
