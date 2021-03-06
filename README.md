# EmpCorrect_App
Use empirical line correction method to convert mosaic image to at-surface reflectance image
# Changelog
===== 27/1/2021 =====
- Add the compatibility with new gdal import method  
  
===== 13/4/2020 =====
- Add the set_draggable attribute to legend to support new matplotlib API  
  
===== 14/9/2017 =====
- Change ROI window's title to DN and reflectance value
- Disable opencv QT context menu when right click
- Make "Original" window automatically appear at the centre of screen, and resize to 2/3 screen size
- Add support to numpad Enter in _Reflectance_ input box  
- Add the function to adjust display brightness (0-300%)  
  
===== 11/9/2017 =====  
- Remove executable since python crash every time when I try to build it 

=====  6/7/2017 ===== 
- The lastest source code assigns QT5Agg as the default backend.  
- Replace os.system with subprocess.run to execute exiftool.  
- Correct Syntax to copy XMP metadata

# This is my first GUI application so that I just take any modules for quick development.
# Any Suggestion for improvement is welcome

Note: The script is coded in Python 3.6 environment on Windows 10 64-bit, and only tested on Windows and Linux (Ubuntu 16.04 with Python 3.5). The script may not work on other platform.  
Please download exiftool from https://sno.phy.queensu.ca/~phil/exiftool/  
 
__Dependency: numpy, scipy, opencv, gdal, matplotlib, PyQt5__  
For Windows users: install modules downloaded from http://www.lfd.uci.edu/~gohlke/pythonlibs/ if pip doesn't work.  

# Usage:
Double click the script to execute the application. Otherwise, use _python3_ in command line (or _python_ if you only install python3 in your computer)   
Use mode 1 to zoom with mouse drag (right click to reset view), and mode 0 to draw polygon to specify pixels (right click to finish drawing).  
After drawing the area of interest, input the at-surface reflectance value for those pixels.  
When finish, press "ESC" and it will generate the linear regression graph.  
If you satisfy the result, close the graph and confirm to continue. The application will apply the linear equation to calculate the at-surface reflectance image.  
