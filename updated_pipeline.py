#import everything

import numpy as np
from astropy.io import fits
from astropy import stats
from PIL import Image
from scipy.ndimage import shift
from astropy.visualization import ZScaleInterval
import os
import glob
import scipy
import re
import matplotlib.pyplot as plt
import shutil
import skimage as ski
import re

#sort
def sort_ims(data_path):
    #this sorts images
    '''
    Sorts the images into bias, flat, and science, and then into filters and objects.
    Input:
        data_path (string): where to find the data
    '''
    #get images
    print('getting images...')

    all_ims = glob.glob('*\\*.*.fits')
    
    print(len(all_ims))

    #check for dirs
    if not os.path.isdir('bias_frames'):
        os.mkdir('bias_frames')
    if not os.path.isdir('flat_frames'):
        os.mkdir('flat_frames')
    if not os.path.isdir('science_frames'):
        os.mkdir('science_frames')
    
    #sort images
    for im in all_ims:
        #print(im)
        #get info
        #"C:\Users\mynam\Documents\school\div ii sem 4\obs tech 2\perkins_data\20250109\20250109\20250109.001.fits"
        try:
            im_data = fits.getdata(im)
        except:
            print(im)
            continue
        im_head = fits.getheader(im)
        im_type = im_head['OBSTYPE']
        im_filt = im_head['FILTNME3']
        im_name = im_head['OBJECT']
        im_name_no_spaces = ''.join(im_name.split(' ')).lower()

        #test images
        if im_filt == 'UNKNOWN' or im_filt == 'OPEN':
            #print('bad')
            continue
        
        #bias
        if im_type == 'BIAS':
            #print('bias')
            shutil.copy(im,'bias_frames')
        #flats
        elif im_type == 'FLAT':
            os.chdir('flat_frames')
            if not os.path.isdir(im_filt):
                os.mkdir(im_filt)
            os.chdir('..')
            #print(im_filt+' flat')
            shutil.copy(im,'flat_frames\\'+im_filt)
        #science
        elif im_type == 'OBJECT':
            os.chdir('science_frames')
            if im_name == '':
                continue
            if not os.path.isdir(im_name_no_spaces):
                os.mkdir(im_name_no_spaces)
            if not os.path.isdir(im_name_no_spaces+'\\'+im_filt):
                os.mkdir(im_name_no_spaces+'\\'+im_filt)
            os.chdir('..')
            #print(os.getcwd())
            #print(im_filt+' science')
            shutil.copy(im,'science_frames\\'+im_name_no_spaces+'\\'+im_filt)

        
    
    return

#median combine
def med_combine(file_list,file_path):
    '''
    Median combines all files into one master image and saves to path.
    Inputs:
        file_list (array): list of files to combine
        file_path (string): path to where to save file
    Ouputs:
        med_frame (array): master image
    '''
    
    #print('this will do median combine')
    
    n = len(file_list)

    first_frame_data = fits.getdata(file_list[0])
    first_frame_header = fits.getheader(file_list[0])

    imsize_y,imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y,imsize_x,n))

    for ii in range(0,n):
        im = fits.getdata(file_list[ii])
        fits_stack[:,:,ii] = im

    med_frame = np.nanmedian(fits_stack,axis=2)

    
    fits.writeto(file_path,med_frame,first_frame_header,overwrite=True)

    return med_frame

#flat field generation
def flat_field_gen(flat_files,master_bias):
    '''
    Combines the flat images into one master flat.
    Inputs:
        flat_files (array): list of flat field frames
        master_bias (array): master bias
    Ouputs:
        all_master_flats (array): master flat for each filter
    '''

    #print('this will do flat field generation')

    
    #norm combine for each filter
    all_master_flats = []
    os.mkdir('master_flats')
    for flats in flat_files:
        n = len(flats)
        print(n)

        first_frame_data = fits.getdata(flats[0])
        first_frame_header = fits.getheader(flats[0])
        exp_time = first_frame_header['EXPTIME']
        filt = first_frame_header['FILTNME3']
        imsize_y,imsize_x = first_frame_data.shape
        fits_stack = np.zeros((imsize_y,imsize_x,n))

        for ii in range(0,n):
            im = fits.getdata(flats[ii])
            raw_im = (im-master_bias)/exp_time
            norm_im = raw_im/np.nanmedian(raw_im)
            fits_stack[:,:,ii] = norm_im
            fits.writeto('flat_frames\\'+filt+'\\fb_'+os.path.basename(flats[ii]),norm_im,header=first_frame_header,overwrite=True)

        med_frame = np.nanmedian(fits_stack,axis=2)
        day = flats[0].split('\\')[-1].split('.')[0]
        file_path = 'master_flats\\'+day+'_master_'+first_frame_header['FILTNME3']+'_flat.fits'
        print(file_path)
        all_master_flats.append(file_path)
        fits.writeto(file_path,med_frame,header=first_frame_header,overwrite=True)
    return all_master_flats

#fringe
def create_fringe(im_list,master_bias,file_name):
    '''
    creates a fringe frame
    inputs:
        im_list (array): list of images
        master_bias (fits array): fits data for the master bias
    output:
        fringe_frame (fits array): fits data for the fringe frame
    '''
    n = len(im_list)
    first_frame_data = fits.getdata(im_list[0])
    first_frame_header = fits.getheader(im_list[0])
    imsize_y,imsize_x = first_frame_data.shape
    
   
    master_flats = glob.glob('master_flats\\*.fits')
    print(len(master_flats))

    new_fits_stack = np.zeros((imsize_y,imsize_x,int(n/50)+1))

    for j in range(0,int(n/50)):
        fits_stack = np.zeros((imsize_y,imsize_x,50))
        for ii in range(j*50,(j+1)*50):
            head = fits.getheader(im_list[ii])
            exp_time = head['EXPTIME']
            filt = head['FILTNME3']
            im_data = fits.getdata(im_list[ii])
            sci_day = im_list[ii].split('\\')[-1].split('.')[0]
            #print(sci_day)

            diff_list = [abs(int(sci_day)-int(a.split('\\')[-1].split('_')[0])) if fits.getheader(a)['FILTNME3'] == filt else 100 for a in master_flats]
            min_ind = diff_list.index(min(diff_list))
            master_flat = fits.getdata(master_flats[min_ind])
            #print('using flat frame from',int(master_flats[min_ind].split('\\')[-1].split('_')[0]))

            bias_sub = im_data-master_bias
            raw_im = bias_sub/exp_time
            norm_im = raw_im/np.nanmedian(raw_im)
            flat_div = norm_im/master_flat
            fits_stack[:,:,ii-j*50] = flat_div

        med = np.nanmedian(fits_stack,axis=2)
        new_fits_stack[:,:,j] = med
    
    fits_stack = np.zeros((imsize_y,imsize_x,(n%50)))
    for ii in range(n-n%50,n):
        head = fits.getheader(im_list[ii])
        exp_time = head['EXPTIME']
        filt = head['FILTNME3']
        im_data = fits.getdata(im_list[ii])
        sci_day = im_list[ii].split('\\')[-1].split('.')[0]

        diff_list = [abs(int(sci_day)-int(a.split('\\')[-1].split('_')[0])) if fits.getheader(a)['FILTNME3'] == filt else 100 for a in master_flats]
        min_ind = diff_list.index(min(diff_list))
        master_flat = fits.getdata(master_flats[min_ind])
        #print('using flat frame from',int(master_flats[min_ind].split('\\')[-1].split('_')[0]))

        bias_sub = im_data-master_bias
        raw_im = bias_sub/exp_time
        norm_im = raw_im/np.nanmedian(raw_im)
        flat_div = norm_im/master_flat
        fits_stack[:,:,ii-(n-n%50)] = flat_div

    med = np.nanmedian(fits_stack,axis=2)
    new_fits_stack[:,:,-1] = med


    new_header = fits.Header()
    new_header['OBJTYPE'] = 'fringe'

    fringe_frame = np.nanmedian(new_fits_stack,axis=2)

    fits.writeto(file_name,fringe_frame,header=new_header,overwrite=True)
    
    return fringe_frame


#bias subtract, flat field divide
def subtract(im_path,master_bias,master_flats,new_path):
    #print('this will subtract')
    '''
    Subtracts bias and flat from a science image.
    Inputs:
        im_path (string): path to science image
        master_bias (array): master bias
        master_flats (array): master flats
        new_path (string): path to new file location
    '''

    #get image info
    science_im = fits.getdata(im_path)
    science_head = fits.getheader(im_path)
    filt = science_head['FILTNME3']
    sci_day = im_path.split('\\')[-1].split('.')[0]
    im_name = science_head['OBJECT']
    im_name_no_spaces = ''.join(im_name.split(' ')).lower()

    #subtract bias
    bias_sub = science_im - master_bias
    if not os.path.isdir('increments'):
        os.mkdir('increments')
    if not os.path.isdir('increments\\'+im_name_no_spaces):
        os.mkdir('increments\\'+im_name_no_spaces)
    fits.writeto('increments\\'+im_name_no_spaces+'\\bias_sub_'+os.path.basename(im_path),bias_sub,header=science_head,overwrite=True)
    
    #divide flat
    diff_list = [abs(int(sci_day)-int(a.split('\\')[-1].split('_')[0])) if fits.getheader(a)['FILTNME3'] == filt else 100 for a in master_flats]
    #print(diff_list)
    min_ind = diff_list.index(min(diff_list))
    #print(min_ind)
    master_flat = fits.getdata(master_flats[min_ind])

    flat_div = bias_sub/master_flat
    fits.writeto('increments\\'+name+'\\flat_div_'+os.path.basename(im_path),flat_div,header=science_head,overwrite=True)

    #divide fringe
    fringe_im = fits.getdata('fringe_'+filt+'.fits')
    fringe_div = flat_div/fringe_im
    fits.writeto('increments\\'+name+'\\fringe_div_'+os.path.basename(im_path),fringe_div,header=science_head,overwrite=True)

    #final
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    if not os.path.isdir(new_path+'\\'+filt):
        os.mkdir(new_path+'\\'+filt)
    file_path = new_path+'\\'+filt+'\\fb_'+os.path.basename(im_path)
    fits.writeto(file_path,fringe_div,science_head,overwrite=True)

    return

#align
def align(ref_im,im):
    '''
    runs skimage package on substamps of images
    inputs:
        ref_im (image array): image to align to
        im (image array): image to align
    output:
        offs ([float]): calculated offsets to shift image by
    '''
    im_data = fits.getdata(im)
    ref_data = fits.getdata(ref_im)

    im_gray = im_data.astype(float)
    ref_gray = ref_data.astype(float)

    im_gray[np.isnan(im_gray)] = 0
    im_gray[np.isinf(im_gray)] = 0
    ref_gray[np.isnan(ref_gray)] = 0
    ref_gray[np.isinf(ref_gray)] = 0

    im_y,im_x = im_gray.shape

    sub_im = im_gray[int(im_y/2-200):int(im_y/2+200),int(im_x/2-200):int(im_x/2+200)]
    sub_ref = ref_gray[int(im_y/2-200):int(im_y/2+200),int(im_x/2-200):int(im_x/2+200)]
    
    return ski.registration.phase_cross_correlation(sub_ref,sub_im,upsample_factor=5)[0]


#image registration
def im_reg(file_list,offs_list,pad,new_path):
    #print('this will stack all images')
    '''
    aligns and stacks images
    inputs:
        file_list (array): all images to stack
        offs_list (array): list of offsets
        pad (int): amount to pad images by
        new_path (string): where to save final image
    output:
        stack_im (fits array): stacked image
    '''
    reg_ims = []
    file_header = fits.getheader(file_list[0])
    im_name = file_header['OBJECT']
    im_name_no_spaces = ''.join(im_name.split(' ')).lower()

    for i, (im_path, (y_off,x_off)) in enumerate(zip(file_list,offs_list)):
        hdr = fits.getheader(im_path)
        with fits.open(im_path) as hdul:
            im_data = hdul[0].data

        im_data[np.isinf(im_data)] = 0.0
        im_data[np.isnan(im_data)] = 0.0

        pad_im = np.pad(im_data,pad,mode='constant',constant_values = 0)

        shift_im = shift(pad_im,[y_off,x_off])
        fits.writeto('increments\\'+im_name_no_spaces+'\\shifted_'+os.path.basename(im_path),shift_im,hdr,overwrite=True)

        if i%10 == 0:
            print('done shifting',i,'images')

        reg_ims.append(shift_im)

    print('done shifting all images')

    stack_im = np.median(reg_ims,axis=0)
    fits.writeto(new_path,stack_im,file_header,overwrite=True)

    return stack_im
    

#reduction (putting it all together)
def reduction(data_path,name):
    #print('this will run through all functions and reduce data')
    
    os.chdir(data_path)

    #sorting images
    try:
        with open("done_sorting.txt", "x") as f:
            f.write("Sorting images...")
            print('Sorting images...')

            sort_ims(data_path)
            f.close()
    except FileExistsError:
        print("Already sorted.")
    
    #get bias
    bias_frames = glob.glob('bias_frames\\*.fits')


    #median combine bias
    if os.path.isfile('master_bias.fits'):
        print('Already made master bias.')
        master_bias = fits.getdata('master_bias.fits')
    else:
        print('Making master bias...')
        master_bias = med_combine(bias_frames,'master_bias.fits')

    #get flats
    flat_frames = []
    filt_list = os.listdir('flat_frames')
    
    list_of_days = []

    for im in glob.glob('science_frames\\'+name+'\\*\\*.*.fits'):
        day = im.split('\\')[3].split('.')[0]
        #print(day)
        list_of_days.append(day)

    list_of_days = set(list_of_days)

    for f in filt_list:
        for day in list_of_days:
            flats = glob.glob('flat_frames\\'+f+'\\'+day+'.*.fits')
            if len(flats) > 0:
                flat_frames.append(flats)

    #generate flat fields
    if os.path.isdir('master_flats'):
        print('Already made master flats.')
        master_flats = glob.glob('master_flats\\*.fits')
    else:
        print('Making master flats...')
        master_flats = flat_field_gen(flat_frames,master_bias)

    print(len(master_flats))

    #get science
    #print(os.getcwd())
    sci_ims = []
    filt_list = os.listdir('science_frames\\'+name)
    #print(filt_list)

    #make fringe frame
    fringe_list = []
    for f in filt_list:
        fringe_scis = glob.glob('science_frames\\*\\'+f+'\\2025*.fits')
        tar_scis = glob.glob('science_frames\\'+name+'\\'+f+'\\*.fits')
        if not os.path.isfile('fringe_'+f+'.fits'):
            print('Making',f,'fringe frame...')
            fringe = create_fringe(fringe_scis,master_bias,'fringe_'+f+'.fits')
            fringe_list.append(fringe)
        else:
            print('Already made',f,'fringe frame.')
        #print(len(scis))
        sci_ims = sci_ims + tar_scis

    #print(len(sci_ims))

    #calibrate
    if os.path.isdir('calibrated\\'+name):
        print('Already calibrated '+name+'.')
    else:
        print('calibrating science images...')
        if not os.path.isdir('calibrated'):
            os.mkdir('calibrated')
        for i,im in enumerate(sci_ims):
            if i%10 == 0:
                print('calibrating the ',str(i),'th image...')
            subtract(im,master_bias,master_flats,'calibrated\\'+name)

    #get target coords in images
    calibrated = glob.glob('calibrated\\'+name+'\\*\\*.fits')
    print(len(calibrated))
    offs_list = []
    if len(calibrated) == 0:
        print('Length zero.')
        return
    #target_ra = input(name+' RA:   ')
    #target_dec = input(name+' Dec:   ')
    print('Getting offsets...')
    ref_im = calibrated[0]
    for im in calibrated:
        print('\nimage:',im)
        offs = align(ref_im,im)
        print(offs)
        #offs_list.append(offs)
        offs_list.append(offs)
    
    print('Done getting offsets.')

    for f in filt_list:
        if not os.path.isdir('final_images'):
            os.mkdir('final_images')
        if os.path.isfile('final_images\\final_stacked_'+name+'_'+f+'.fits'):
            print('Already stacked images.')
        else:
            print('Stacking images...')
            calibrated = glob.glob('calibrated\\'+name+'\\'+f+'\\*.fits')
            print(len(calibrated))
            final_im = im_reg(calibrated,offs_list,10,'final_images\\final_stacked_'+name+'_'+f+'.fits')
            print('Done stacking.')

    return


#main function
if __name__ == '__main__':
    #print('this will run reduction function from terminal')
    datafolder = input('Data Folder Name:  ')
    name = input('Object Name:   ')
    reduction(datafolder,name)