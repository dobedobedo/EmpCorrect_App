#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
The mosaic has actually nodata value at 0.
However, due to some technical issue, it is set to some other value automatically.
Therefore, the nodata value is set manually here.
0 for mosaic
-10000 for reflectnace image
you can change the extraction method by uncomment the ndv sourcecode.
"""
import matplotlib
matplotlib.use('Qt5Agg')


import os
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
import cv2
import gdal
from osgeo import gdal_array
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import font_manager
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FormatStrFormatter

drawing = False
mode = True
roi_corners = []
ix, iy = -1, -1
i = 1
mean_DN = []
reflectance = []
brightness = 1.0

#Read image using GDAL to avoid unusual bands problem by opencv
def Read_Image(filename):
    InputFile = gdal.Open(filename)
    cols = InputFile.RasterXSize
    rows = InputFile.RasterYSize
    channel = InputFile.RasterCount
    GeoTransform = InputFile.GetGeoTransform()
    Projection = InputFile.GetProjection()
    driver = InputFile.GetDriver()
    bands = []
    for band in range(channel):
        bands.append(InputFile.GetRasterBand(band+1))
    #ndv = bands[band].GetNoDataValue() #Get nodata automatically
    ndv = 0   #Set nodata manually to 0, which is the usual situation
    if channel <= 2:
        image = np.zeros((rows,cols), dtype=InputFile.ReadAsArray().dtype)
    else:
        image = np.zeros((rows,cols,channel), dtype=InputFile.ReadAsArray().dtype)
    
    for band in range(channel):
        if channel == 1:
            image = bands[band].ReadAsArray(0,0,cols,rows)
            alpha = None
        elif channel == 2:
            if band != channel-1:
                image = bands[band].ReadAsArray(0,0,cols,rows)
            else:
                alpha = bands[band].ReadAsArray(0,0,cols,rows)
        elif channel == 4:
            if band != channel-1:
                image[:,:,band] = bands[band].ReadAsArray(0,0,cols,rows)
            else:
                alpha = bands[band].ReadAsArray(0,0,cols,rows)
        else:
            image[:,:,band] = bands[band].ReadAsArray(0,0,cols,rows)
            alpha = None
    InputFile = None    
    return image, ndv, alpha, GeoTransform, Projection, driver

#draw polygon or zoom by rectangle depend on the switch trackbar
def draw_polygon(event,x,y,flags,param):
    global drawing, mode, roi_corners, ix, iy, i, temp, temp_zoom, mean_DN, reflectance, brightness
        
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if mode == True:
            if len(roi_corners) == 0:
                roi_corners = [(x,y)]
                temp_zoom = temp.copy()
            else:
                roi_corners.append((x,y))
        else:
            ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            draw_image = temp.copy()
            if mode == True:
                cv2.line(draw_image, roi_corners[-1], (x,y), (0,160,0), 1)
                cv2.imshow("Original", (np.clip(draw_image*brightness, 
                                                np.iinfo(temp.dtype).min, 
                                                np.iinfo(temp.dtype).max)).astype(temp.dtype))
            else:
                cv2.rectangle(draw_image, (ix,iy), (x,y), (0,0,160), -1)
                cv2.addWeighted(draw_image, 0.4, temp, 0.6, 0, draw_image)
                cv2.imshow("Original", (np.clip(draw_image*brightness, 
                                                np.iinfo(temp.dtype).min, 
                                                np.iinfo(temp.dtype).max)).astype(temp.dtype))

    elif event == cv2.EVENT_LBUTTONUP:
        if mode == True:
            if len(roi_corners) > 1:
                if len(roi_corners) >= 3:
                    for pt in range(1,len(roi_corners)-1):
                        error_code = line_check(
                                    roi_corners[pt-1], roi_corners[pt]
                                    , roi_corners[-2], roi_corners[-1])
                        if error_code == 1:
                            drawing = False
                            temp = temp_zoom.copy()
                            cv2.imshow("Original", (np.clip(temp*brightness, 
                                                            np.iinfo(temp.dtype).min, 
                                                            np.iinfo(temp.dtype).max)).astype(temp.dtype))
                            roi_corners = []
                            messagebox.showinfo("Warning",
                                        "Polygon cannot be self-intersected!")
                            break
                        elif error_code == 2:
                            drawing = False
                            temp = temp_zoom.copy()
                            cv2.imshow("Original", (np.clip(temp*brightness, 
                                                            np.iinfo(temp.dtype).min, 
                                                            np.iinfo(temp.dtype).max)).astype(temp.dtype))
                            roi_corners = []
                            messagebox.showinfo("Warning",
                                        "Duplicated edges!")
                            break
                    if error_code == 0:
                        cv2.line(temp, roi_corners[-2], 
                                 roi_corners[-1], (0,160,0), 1)
                        cv2.imshow("Original", (np.clip(temp*brightness, 
                                                        np.iinfo(temp.dtype).min, 
                                                        np.iinfo(temp.dtype).max)).astype(temp.dtype))
                else:
                    cv2.line(temp, roi_corners[-2], 
                             roi_corners[-1], (0,160,0), 1)
                    cv2.imshow("Original", (np.clip(temp*brightness, 
                                                    np.iinfo(temp.dtype).min, 
                                                    np.iinfo(temp.dtype).max)).astype(temp.dtype))
        else:
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            x1, y1, w, h = rect
            if w != 0 and h != 0:
                temp = temp[y1:y1+h, x1:x1+w]
                temp_zoom = temp.copy()
            cv2.imshow("Original", (np.clip(temp*brightness, 
                                            np.iinfo(temp.dtype).min, 
                                            np.iinfo(temp.dtype).max)).astype(temp.dtype))
            drawing = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        if mode == True:
            drawing = False
            if len(roi_corners) < 3:
                messagebox.showinfo("Warning",
                                    "You need at least 3 points to form a polygon!")
                temp = temp_zoom.copy()
                cv2.imshow("Original", (np.clip(temp*brightness, 
                                                np.iinfo(temp.dtype).min, 
                                                np.iinfo(temp.dtype).max)).astype(temp.dtype))
                roi_corners = []
            else:
                roi_corners.append(roi_corners[0])
                for pt in range(2,len(roi_corners)-1):
                    error_code = line_check(
                            roi_corners[pt-1], roi_corners[pt]
                            , roi_corners[-2], roi_corners[-1])
                    if error_code == 1:
                        temp = temp_zoom.copy()
                        cv2.imshow("Original", (np.clip(temp*brightness, 
                                                        np.iinfo(temp.dtype).min, 
                                                        np.iinfo(temp.dtype).max)).astype(temp.dtype))
                        roi_corners = []
                        messagebox.showinfo("Warning",
                                    "Polygon cannot be self-intersected!")
                        break
                    elif error_code == 2:
                        temp = temp_zoom.copy()
                        cv2.imshow("Original", (np.clip(temp*brightness, 
                                                        np.iinfo(temp.dtype).min, 
                                                        np.iinfo(temp.dtype).max)).astype(temp.dtype))
                        roi_corners = []
                        messagebox.showinfo("Warning",
                                    "Duplicated edges!")
                        break
                if error_code == 0:
                    cv2.line(temp, roi_corners[-2], roi_corners[-1], (0,160,0), 1)
                    cv2.imshow("Original", (np.clip(temp*brightness, 
                                                    np.iinfo(temp.dtype).min, 
                                                    np.iinfo(temp.dtype).max)).astype(temp.dtype))
                    temp = temp_zoom.copy()
                    mask = create_mask(temp.shape, roi_corners, temp.dtype)
                    masked_image = cv2.bitwise_and(temp, mask)
                    mean_DN.append(ma.array(temp, mask=np.invert(mask)).mean())
                    reflectance.append(Ref_inputBox())
                    cv2.namedWindow("ROI {}".format(i),cv2.WINDOW_NORMAL)
                    cv2.imshow("ROI {}".format(i), (np.clip(masked_image*brightness, 
                               np.iinfo(masked_image.dtype).min, 
                               np.iinfo(masked_image.dtype).max)).astype(masked_image.dtype))
                    roi_corners = []
                    i = i + 1
        else:
            temp = image_used.copy()
            temp_zoom = temp.copy()
            cv2.imshow("Original", (np.clip(temp*brightness, 
                                            np.iinfo(temp.dtype).min, 
                                            np.iinfo(temp.dtype).max)).astype(temp.dtype))

def create_mask(shape, roi_corners, img_type):
    mask = np.zeros(shape, dtype=img_type)
    fill_value = np.iinfo(img_type).max
    roi = np.array([roi_corners], dtype=np.int32)
    
    if len(shape) < 3:
        ignore_mask_colour = (fill_value,)
    else:
        ignore_mask_colour = (fill_value,)*shape[-1]
    
    cv2.fillPoly(mask, roi, ignore_mask_colour)
    return mask

def mode_switch(x):
    global mode, roi_corners, drawing, brightness
    drawing = False
    temp = temp_zoom.copy()
    cv2.imshow("Original", (np.clip(temp*brightness, 
                                    np.iinfo(temp.dtype).min, 
                                    np.iinfo(temp.dtype).max)).astype(temp.dtype))
    if x == 0:
        mode = True
    else:
        mode = False
    roi_corners = []

def adjust_brightness(x):
    global brightness
    brightness = x/100.0
    cv2.imshow("Original", (np.clip(temp*brightness, 
                                    np.iinfo(temp.dtype).min, 
                                    np.iinfo(temp.dtype).max)).astype(temp.dtype))

def line_check(pt1, pt2, pt3, pt4):
    error_code = 0
    line1 = [pt1, pt2]
    line2 = [pt3, pt4]
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    #calculate line intersection using Cramer's rule
    def det(a,b):
        return a[0]*b[1] - a[1]*b[0]
    div = det(xdiff, ydiff)
    d = (det(*line1), det(*line2))
    try:
        x = det(d, xdiff)/div
        y = det(d, ydiff)/div
        #check whether the intersections are within segments
        if not ((x < max(min(pt1[0], pt2[0]), min(pt3[0], pt4[0])) or
            x > min(max(pt1[0], pt2[0]), max(pt3[0], pt4[0]))) and 
            (y < max(min(pt1[1], pt2[1]), min(pt3[1], pt4[1])) or
            y > min(max(pt1[1], pt2[1]), max(pt3[1], pt4[1])))):
            if x != pt3[0] and y != pt3[1]:
                error_code = 1
        return error_code
    
    #except condition if segments are parallel or colinear
    except ZeroDivisionError:
        try:
            l1a = (line1[1][1]-line1[0][1])/float((line1[1][0]-line1[0][0]))
            l1b = line1[0][1] - l1a*line1[0][0]
            l2a = (line2[1][1]-line2[0][1])/float((line2[1][0]-line2[0][0]))
            l2b = line2[0][1] - l2a*line2[0][0]
        #except condition if segments are vertical lines
        except ZeroDivisionError:
            l1b = line1[0][0]
            l2b = line2[0][0]
        if l1b == l2b:
            if not (pt4[1] < max(min(pt1[1], pt2[1]), min(pt3[1], pt4[1])) or
            pt4[1] > min(max(pt1[1], pt2[1]), max(pt3[1], pt4[1]))):
                error_code = 2
        return error_code

def Ref_inputBox():
    class popupWindow(tk.Tk):
        def __init__(self):
            tk.Tk.__init__(self)
            self.resizable(width=False, height=False)
            self.title("")
            self.l=tk.Label(self,text="Reflectance")
            self.l.pack()
            self.e=tk.Entry(self, width=30)
            self.e.bind('<Return>', self.cleanup)
            self.e.bind('<KP_Enter>', self.cleanup)
            self.e.pack()
            self.b=tk.Button(self,text="Ok", width=40, height=2)
            self.b.bind('<Button-1>', self.cleanup)
            self.b.pack()
            # Make popup window at the centre
            self.update_idletasks()
            w = self.winfo_screenwidth()
            h = self.winfo_screenheight()
            size = tuple(int(_) for _ in self.geometry().split('+')[0].split('x'))
            x = w/2 - size[0]/2
            y = h/2 - size[1]/2
            self.geometry("%dx%d+%d+%d" % (size + (x, y)))
        def cleanup(self, event):
            try:
                self.value=self.e.get()
                float(self.value)
                self.quit()
            except ValueError:
                if len(self.value) > 0:
                    messagebox.showerror("Warning!", 
                                         "Input must be number!")
                else:
                    messagebox.showerror("Warning!",
                                        "Input cannot be blank!")
                self.e.delete(0, 'end')
    m=popupWindow()
    m.mainloop()
    m.destroy()
    return eval(m.value)

def Plot_Line(DN, reflectance):
    
    #define linear regression function
    def linear_func(x, p1, p2):
        return p1*x+p2
    
    def R_squared(data, residuals):
        #calculate linear residual's sum of squares
        ssres = sum(residuals**2)
        #calculate original residual's total sum of squares
        sstol = sum((data-np.mean(data))**2)
        #calculate and return coefficient of determination
        return 1 - ssres/sstol
    
    #calculate empirical line equation's coefficients and covariances
    popt, pcov = curve_fit(linear_func, DN, reflectance)
    
    #calculate coefficiant of determination R2
    residuals = reflectance - linear_func(DN, popt[0], popt[1])
    R2 = R_squared(reflectance, residuals)
    
    #Make the empirical line have wider range
    linex = np.linspace(min(DN)*0.5, max(DN)*1.5, len(DN)*2)
    liney = linear_func(linex, popt[0], popt[1])
    
    #Set fonttype of the figure
    system_font = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    fontlist = ["Times New Roman", "Helvetica", "DejaVu Serif"]
    default_font = ''
    for selected_font in fontlist:
        for font_with_path in system_font:
            font = font_manager.FontProperties(fname=font_with_path).get_name()
            if selected_font in font:
                default_font = selected_font
                break
    rc("font", family = default_font)
    
    #Configure figure
    plt.figure(1)
    plt.gca().set_ylim(min(reflectance)*0.9,max(reflectance)*1.1)
    plt.gca().set_xlim(min(DN)*0.9, max(DN)*1.1)
    plt.subplot(111).spines['bottom'].set_linewidth(2)
    plt.subplot(111).spines['left'].set_linewidth(2)
    plt.subplot(111).xaxis.set_major_locator(AutoLocator())
    plt.subplot(111).xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.subplot(111).xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel("Digital Number (DN)", size = 20, labelpad = 10)
    plt.xticks(size = 14)
    plt.ylabel("Reflectance", size = 20, labelpad = 10)
    plt.yticks(size = 14)
    plt.title("DN versus reflectance", size = 26).set_y(1.05)
    plt.grid(which='major', alpha=0.6, ls='-')
    plt.grid(which='minor', alpha=0.3, ls='--')
    
    #Plot dsta
    plt.plot(DN, reflectance, 'bo', label="Sample Data")
    plt.plot(linex, liney, 'k-', label="Empirical Line")
    plt.annotate("$y=({:.2e})x{:+.2f}$\n$R^2={:.2f}$".format(popt[0], popt[1], R2), 
                 xy=(0.05, 0.7), xytext=(0, 0), xycoords=('axes fraction', 'axes fraction'),
                 textcoords='offset points',
                 verticalalignment='bottom', horizontalalignment='left', 
                 fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.legend(loc='lower center',bbox_to_anchor = (0.76, 0.02), fancybox = True, 
               fontsize=16, markerscale=1, ncol = 1).draggable()

    plt.show()
    return popt
    
gdal.AllRegister()
tk.Tk().withdraw()
'''
Fullfilename=askopenfilename(filetypes=[(
        "Common 8-bit images","*.jpg;*.jpeg;*.png"),("Tiff","*.tif;*.TIF")])
'''
Fullfilename=askopenfilename()
try:
    path, filename = os.path.split(Fullfilename)
    filename, ext = os.path.splitext(filename)
    OutFile = os.path.join(path,filename+'_out.tif')
    image, ndv, alpha, GeoTransform, Projection, driver = \
        Read_Image(os.path.join(path, filename+ext))
    
    if ndv is not None:
        image = ma.masked_values(image,ndv)
        image_used = image.filled(0)
    else:
        image_used = image.copy()
    
    if alpha is not None:
        if alpha.max() != np.iinfo(alpha.dtype).max:
            alpha[alpha==alpha.max()] = np.iinfo(alpha.dtype).max
    if len(image_used.shape) > 2:
        if image_used.shape[2] == 3:
            image_used = image_used[...,::-1]
    temp = image_used.copy()
    temp_zoom = temp.copy()
    messagebox.showinfo(
            "Control tips", 
            "Use scroll bar to change mode:\n\n"
            "mode 0: draw polygon for regions of interest, right click to finish\n"
            "mode 1: zoom with rectangle, right click to reset extent\n\n"
            "When finish, press ESC to continue.")
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    
    cv2.createTrackbar("mode", "Original", 0, 1, mode_switch)
    cv2.createTrackbar("brightness%", "Original", 100, 300, adjust_brightness)
    
    cv2.setMouseCallback("Original", draw_polygon)
    
    while cv2.getWindowProperty("Original", 0) >= 0:
        cv2.imshow("Original", temp)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        cv2.getTrackbarPos("mode", "Original")
        cv2.getTrackbarPos("brightness%", "Original")
    
    cv2.destroyAllWindows()
    
    if len(mean_DN) == len(reflectance) > 1:
        messagebox.showinfo("Note", 
                            "Generating linear regression... \
                            \nAfter finish, close the window to continue.")
        popt = Plot_Line(np.array(mean_DN), np.array(reflectance))
        User_confirm = messagebox.askquestion(
                "Confirmation", 
                "Are you sure to apply the equation to manipulate the image?")
        if User_confirm == 'yes':
            image = image * popt[0] + popt[1]
            
            #Reduce precision to decrease file size
            image = image.astype(np.float32)
            
            if ndv is not None:
                #image = image.filled(ndv)
                image = image.filled(-10000) #nodata is set to -10000 for reflectance image
                
            if len(image.shape) > 2:
                band = image.shape[2]
                if image.shape[2] == 3:
                    image = image[...,::-1]
            else:
                band = 1
            
            if alpha is not None:
                image = np.stack((image,alpha), axis=-1)
            Type = gdal_array.NumericTypeCodeToGDALTypeCode(image.dtype.type)    
            OutImage = driver.Create(OutFile, 
                                     image.shape[1], image.shape[0], band, Type)
            if band == 1:
                try:
                    OutImage.GetRasterBand(1).WriteArray(image[:,:])
                except:
                    OutImage.GetRasterBand(1).WriteArray(image[:,:,0])
                if ndv is not None:
                        #OutImage.GetRasterBand(1).SetNoDataValue(ndv)
                        OutImage.GetRasterBand(1).SetNoDataValue(-10000)
            else:
                for i in range(band):
                    OutImage.GetRasterBand(i+1).WriteArray(image[:,:,i])
                    if ndv is not None:
                        #OutImage.GetRasterBand(i+1).SetNoDataValue(ndv)
                        OutImage.GetRasterBand(i+1).SetNoDataValue(-10000)
                        
            OutImage.SetGeoTransform(GeoTransform)
            OutImage.SetProjection(Projection)
        
            OutImage = None
            os.system(
                    "exiftool -tagsFromFile \"{Src}\" -all:all "
                    "-tagsFromFile \"{Src}\" -xmp:all \"{Dst}\" && "
                    "exiftool -delete_original! \"{Dst}\"".format(
                            Src=Fullfilename, 
                            Dst=OutFile))
            messagebox.showinfo("Done!", 
                                "Image is saved as {}_out.tif".format(filename))
        else:
            messagebox.showinfo("Discard!", 
                                "No output image is created.")
    else:
        messagebox.showinfo("Discard!", 
                                "Too few sample points. No output image is created.")

except:
    messagebox.showinfo("Discard!", "No image is selected.")
