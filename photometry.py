#photometry calculations

#import stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os.path
import glob

# Newer packages:
# You may need to install them via pip
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error
import astropy.stats as stat

from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder

from IPython.display import Image

from astropy.wcs import WCS
from astropy.time import Time

#warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

from DiffPhot import *

def bg_error_estimate(fitsfile):
    """
    Estimates background error in a fits file.
    Input:
        fitsfile (string): path to fits file
    Output:
        error_image (array): array of pixel errors
    """
    fitsdata = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    filt = hdr['FILTNME3']
    
    # What is happening in the next step? Read the docstring for sigma_clip.
    # Answer: removes data a certain number of standard deviations from the mean (ie keeps only the centered data)
    filtered_data = sigma_clip(fitsdata, sigma=3.,copy=False)
    
    # Summarize the following steps. Answer:
    # Fills empty indicies in array with Nans
    # Makes array were each pixel is square root of previous pixel of last array
    # Sets NaN values equal to the median of the non-NaN values
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    print("Writing the background-only error image: ", fitsfile.split('.')[0]+'.'+filt+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+'.'+filt+"_bgerror.fit", bkg_error, hdr, overwrite=True)
    
    effective_gain = 1.4 # electrons per ADU
    
    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)  
    
    print("Writing the total error image: ", fitsfile.split('.')[0]+'.'+filt+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+'.'+filt+"_error.fit", error_image, hdr, overwrite=True)
    
    return error_image

# Star extraction function 
def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    Takes an image and outputs a list of all star coordinates above the defined threshold.
    Inputs:
        fitsfile (string): path to fits file
        nsigma_value (int): number of standard deviations away to use for a threshold
        fwhm_value (float): width of aperture for locating stars
    Outputs:
        xpos, ypos (arrays): arrays of centroid x and y positions of stars
    """
    
    # First, check if the region file exists yet, so it doesn't get overwritten
    regionfile = fitsfile.split(".")[0] + ".reg"
    
    '''
    if os.path.exists(regionfile) == True:
        print(regionfile, "already exists in this directory. Rename or remove the .reg file and run again.")
        return
    '''
    # get data of fitsfile
    image = fits.getdata(fitsfile)
    '''
    cropx = 2000
    cropy = 2000
    print(image.shape)
    startx = image.shape[0]//2-cropx//2
    print(startx)
    starty = image.shape[1]//2-cropy//2
    image = image[startx:startx+cropx,starty:starty+cropy]
    '''
    # *** Measure the median absolute standard deviation of the image: ***
    bkg_sigma = mad_std(image[~np.isnan(image)])

    # *** Define the parameters for DAOStarFinder ***
    daofind = DAOStarFinder(fwhm=fwhm_value, threshold=nsigma_value*bkg_sigma)
    
    # Apply DAOStarFinder to the image
    sources = daofind(image)
    nstars = len(sources)
    print("Number of stars found in ",fitsfile,":", nstars)
    
    # Define arrays of x-position and y-position
    xpos = np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])
    
    # Write the positions to a .reg file based on the input file name
    f = open(regionfile, 'w') 
    for i in range(0,len(xpos)):
        f.write('circle '+str(xpos[i])+' '+str(ypos[i])+' '+str(fwhm_value)+'\n')
    f.close()
    print("Wrote ", regionfile)
    return xpos, ypos # Return the x and y positions of each star as variables
    
# Photometry function
def measurePhotometry(fitsfile, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array):
    """
    Shows all stars, their values, and the respective total error bars.
    Inputs:
        fitsfile (string): fits file
        star_xpos (float): x coordinate for star
        star_ypos (float): y coordinate for star
        aperture_radius (float): radius of aperture around the star
        sky_inner (float): radius of inner annulus circle
        sky_outer (float): radius of outer annulus circle
        error_array (array): array of errors 
    Output:
        phot_table (array): table of all star attributes
    """
    # *** Read in the data from the fits file:
    image = fits.getdata(fitsfile)
    image[np.isnan(image)] = 0
    image[np.isinf(image)] = 0
    
    star_pos = np.vstack([star_xpos, star_ypos]).T
    
    starapertures = CircularAperture(star_pos,r = aperture_radius)
    skyannuli = CircularAnnulus(star_pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    # What is new about the way we're calling aperture_photometry?
    # This is accounting for the annulus and including error
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
    
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_med = np.median(phot_table['aperture_sum_1'])
    
    bkg_starap_sum = bkg_mean * starapertures.area
    #print(bkg_starap_sum)
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    #print(final_sum)
    phot_table['bg_subtracted_star_counts'] = [a if a >= 0 else 0.0001 for a in final_sum]
    
    # Calculating mean and sum error for the background using annuli area and aperture area
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err = bkg_mean_err * starapertures.area
    
    # Combining errors using error propagation
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table

def do_photo(data_path,name,filt):
    '''
    this will do photometry calculations on a fits file
    inputs:
        data_path (string): path to the image
        name (string): object name
        filt (string): filter
    '''
    os.chdir(data_path)
    im_name = 'final_stacked_'+name+'_'+filt+'.fits'

    file_path = 'science_frames\\'+name+'\\'+filt+'\\*.*.fits'

    im_list = glob.glob(file_path)  
    print(f"Number of images found: {len(im_list)}")

    im_list.sort()

    # This will be the "reference image" used later. You can leave it as-is.
    reference_image = im_list[1]

    target = name

    xpos_target, ypos_target = 1056, 1008

    # Parameters for aperture photometry:
    aprad = 20     # aperture radius (pixels)
    skybuff = 25   # sky annulus inner radius (pixels)
    skywidth = 30  # sky annulus outer radius (pixels)

    # Parameters for star-finding sensitivity:
    nsigma = 10.     # detection threshold in sigma 
    FWHM = 15.       # full width at half-max (pixels)

    # Run StarFind on the reference image to extract its stars.
    xpos, ypos, Nstars = StarFind(reference_image, FWHM, nsigma, aprad, skywidth)

    # Run this to get the index corresponding to the target star.
    seps = np.array([ np.sqrt((xpos[i]-xpos_target)**2 + (ypos[i]-ypos_target)**2) for i in range(Nstars) ])
    idx_target = np.where(seps == min(seps))[0][0]
    print("Target star is this one: ",idx_target)

    # Measure the photometry:
    times_raw, Photometry_initial = doPhotometry(im_list, reference_image, xpos, ypos, aprad, skybuff, skywidth, timekey="DATE-OBS", verbose=0)

    # Convert times_raw to a new variable that Python can manipulate as a Time object:
    times = Time(times_raw, format='isot', scale='utc')

    # Measure the uncertainties on the photometric measurements:
    ePhotometry = doPhotometryError(im_list, xpos, ypos, aprad, skybuff, skywidth, Photometry_initial, manual=False, xboxcorner=2000, yboxcorner=2000, boxsize=200)

    
    Photometry, cPhotometry = detrend(Photometry_initial, ePhotometry, Nstars, report=False)

    # Use the comp_num= argument to adjust the number of comparison stars chosen.
    least_variable_stars = findComparisonStars(Photometry, cPhotometry, plot=True, comp_num=9, idx_target=idx_target)   
    dPhotometry, edPhotometry, tePhotometry  = runDifferentialPhotometry(Photometry, ePhotometry, Nstars, least_variable_stars)

    #plot
    plt.figure(figsize=(8,6))

    # Here is an example of how to access a single star's photometry across all images:
    plt.plot(times.decimalyear, Photometry_initial[:,idx_target], '.', label=name)

    plt.legend(loc='best')
    plt.xlabel("time")  # fill this in!
    plt.ylabel("flux")  # fill this in!
    plt.gcf().autofmt_xdate()  # this turns the x-ticks diagonal
    plt.show()

#main function
if __name__ == '__main__':
    #print('this will run reduction function from terminal')
    datafolder = input('Data Folder Name:  ')
    name = input('Object Name:   ')
    filt = input('Filter:   ')
    do_photo(datafolder,name,filt)