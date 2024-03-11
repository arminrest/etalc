#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:08:01 2023

@author: arest
"""

import os,sys,re,copy,shutil
import numpy as np
import glob
import argparse

from astropy.io import fits,ascii
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import bitmask
import pandas as pd

from pdastro import pdastroclass,pdastrostatsclass,unique,AandB
import matplotlib.pyplot as plt

photcode2filter = {
    0x12:'u',
    0x13:'g',
    0x14:'r',
    0x15:'i',
    0x16:'z',
    0x17:'y'
    }

def addlink2string(s,link):
    return('<a href="%s">%s</a>' % (link,s))

def imagestring4web(imagename, width=None, height=None):
    imstring = '<img src="%s"' % imagename
    if height != None:
        imstring += f' height={height}'
        #if isinstance(height,int): height = str(height)
        #imstring += 'height=%s' % height
    if width != None:
        imstring += f' width={width}'
        #if isinstance(width,int): width = str(width)
        #imstring += 'width=%s' % width
    imstring +='>'
    return(imstring)


def initplot(nrows=1, ncols=1, 
             xfigsize4subplot=7, 
             yfigsize4subplot=5, 
             **kwargs):
    sp=[]
    xfigsize=xfigsize4subplot*ncols
    yfigsize=yfigsize4subplot*nrows
    fig = plt.figure(figsize=(xfigsize,yfigsize))
    counter=1
    for row in range(nrows):
        for col in range(ncols):
            sp.append(plt.subplot(nrows, ncols, counter,**kwargs))
            counter+=1

    for i in range(len(sp)):
        plt.setp(sp[i].get_xticklabels(),'fontsize',12)
        plt.setp(sp[i].get_yticklabels(),'fontsize',12)
        sp[i].set_xlabel(sp[i].get_xlabel(),fontsize=14)
        sp[i].set_ylabel(sp[i].get_ylabel(),fontsize=14)
        sp[i].set_title(sp[i].get_title(),fontsize=14)

    return(sp,fig)

def plotdata(x, y, dx=None, dy=None, sp=None, fmt='bo', alpha=0.5, color='blue', ecolor='k', elinewidth=None, barsabove = False, capsize=1, logx=False, logy=False):
    if sp == None:
        sp = plt.subplot(111)

    if dx is None and dy is None:
        if logy:
            if logx:
                plot, = sp.loglog(x, y, fmt)
            else:
                 plot, = sp.semilogy(x, y, fmt)
        elif logx:
            plot, = sp.semilogx(x, y, fmt)
        else:
            if barsabove:
                plot, dplot,dummy = sp.errorbar(x, y, alpha=alpha, fmt=fmt, color=color, capsize=capsize, barsabove=barsabove)
            else:
                plot, = sp.plot(x, y, fmt)

        #plot = sp.plot(x, y, fmt)
        return sp, plot, None
    else:
        if logy:
            sp.set_yscale("log", nonposx='clip')
        if logx:
            sp.set_xscale("log", nonposx='clip')

        # For newer matplotlib: 3-tuple
        #plot = sp.semilogy(x, y, xerr=dx, yerr=dy, fmt=fmt, ecolor=ecolor, capsize=capsize)
        plot, dplot,dummy = sp.errorbar(x, y, xerr=dx, yerr=dy, fmt=fmt, ecolor=ecolor, elinewidth=elinewidth, capsize=capsize, barsabove=barsabove)
        # for older matplotlib: 2-tuple
        #plot, dplot = sp.errorbar(x, y, xerr=dx, yerr=dy, fmt=fmt, ecolor=ecolor, capsize=capsize)
        return sp, plot, dplot


class positionclass(pdastroclass):
    """
    class for the table of positions for the lc's
    """
    def __init__(self,verbose=0):
        pdastroclass.__init__(self)
        self.posfilename = None

    def load_posfile(self, posfilename=None,checkxyflag=True,imagename=None, wcshdr=None, forcexyflag=True):
        print('\n### Loading position file!')
        if posfilename is None:
            if self.posfilename is None:
                raise RuntimeError('no position filename specified!')
               
            posfilename = self.posfilename
        self.load(posfilename,verbose=4)
        self.posfilename = posfilename
        
        if checkxyflag and ((not ('x' in self.t.columns)) or forcexyflag):
    
            if wcshdr is None:
                if imagename is None:
                    raise RuntimeError('no wcsheader nor imagename specified, cannot determine x/y!')
                wcshdr = fits.getheader(imagename)
            field_wcs = WCS(wcshdr)
    
            (ixs,coords)=self.radeccols_to_SkyCoord(racol='ra',deccol='dec')
            self.t['x'],self.t['y'] = field_wcs.world_to_pixel(coords)
    
            # We should do better rounding here...
            self.t['x'] = self.t['x'].astype('int')
            self.t['y'] = self.t['y'].astype('int')
    
        if self.verbose: self.write()
        
    def define_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)
        parser.add_argument("field", help="name of the light echo field", type=str)
        parser.add_argument("ccd", help="ddtector number", type=str)
        
        self.define_optional_arguments(parser)
        return(parser)

    def define_optional_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        # default directory for position files
        if 'ETALC_POSDIR' in os.environ:
            posdir = os.environ['ETALC_POSDIR']
        else:
            posdir = None            

        parser.add_argument("--posfilename", default='auto', type=str, help="file of RA DEC positions to generate light curves. if 'auto', then <inputrootdir>/posfiles/<field>a<ccd>.lcpos.txt  (default=%(default)s)")
        parser.add_argument("--posdir", default=posdir, help=" position list directory.   (default=%(default)s)")
        #parser.add_argument('-v','--verbose', default=0, action='count')
        
        return(parser)
    
    def set_posfilename(self,field,ccd,
                        posdir=None,posfilename='auto'):
        self.field = field
        self.ccd = ccd
        self.fieldccdID = f'{self.field}a{self.ccd}'

        if posdir is None: posdir = './poslists'

        # detemine the filname for the positions
        if posfilename is None or posfilename.lower()=='auto':
            posfilenameX = os.path.join(posdir,f'{self.fieldccdID}.lcpos.txt')
            if os.path.isfile(posfilenameX):
                self.posfilename = posfilenameX
            else:
                raise RuntimeError(f'Could not find posfilename {posfilenameX}')
        else:
            if os.path.isfile(posfilename):
                self.posfilename = posfilename
            else:
                posfilenameX = os.path.join(posdir,posfilename)
                if os.path.isfile(posfilenameX):
                    self.posfilename = posfilenameX
                else:
                    raise RuntimeError(f'Could not find posfilename {posfilename}')
        
        # Make sure the posfilename is absolute, i.e., remove '~' etc
        self.posfilename = os.path.abspath(self.posfilename)
        print(f'position file: {self.posfilename}')
        return(0)

class extract_LE_lcclass(pdastrostatsclass):
    """
    class that extracts light echo light curve for a list of positions
    """
    def __init__(self,verbose=0):
        pdastrostatsclass.__init__(self)
        
        self.verbose = 0
        
        self.field = None 
        self.ccd = None
        self.filt = None
        
        self.postable = positionclass()
        
        self.datarootdir = None
        self.diff_dir = None
        #self.diff_imgs = None

        self.tmpl_dir = None
        self.tmpl_name = None
        self.tmpl_expnum = None
        
        self.box_size = None
        self.mask_val = None
 
        self.skipmask = {'missing_dcmp':0x1,
                         'missing_noise':0x2,
                         'pipe_nosuccess':0x4}

        self.imtable = pdastroclass()
        self.fitskeys2imtable_required = ['MJD-OBS','ZPTMAGAV','TZPTMAG','SW_PLTSC','IMSUBD','IMNAME','PHOTCODE']
        self.fitskeys2imtable_optional = ['CONVOL00','PHOTNORM','KSUM00','FSIG00','FSCAT00',
                                          'NSCALO00','DSGDNPCP','DSGDNSIG','NSCALO00','DIFFSUCC',
                                          'FWHM','M5SIGMA','SKYADU']
        self.cols_im2lc = ['imID','filter','CONVOL00','PHOTNORM','KSUM00','FSIG00','FSCAT00','FWHM','M5SIGMA','SKYADU']

        self.outbasename = None
        
        self.webpagelist = pdastroclass()
        

    def format_imtable(self):
        self.imtable.default_formatters = {'KSUM00':'{:.4f}'.format,
                                           'FSIG00':'{:.3f}'.format,
                                           'FSCAT00':'{:.3f}'.format,
                                           'NSCALO00':'{:.2f}'.format,
                                           'DSGDNPCP':'{:.2f}'.format,
                                           'DSGDNSIG':'{:.3f}'.format,
                                           'MJD-OBS':'{:.6f}'.format,
                                           'FWHM':'{:.3f}'.format,
                                           'M5SIGMA':'{:.3f}'.format,
                                           'ZPTMAGAV':'{:.3f}'.format,
                                           'SKYADU':'{:.2f}'.format,
                                           'SW_PLTSC':'{:.4f}'.format,
                                           'PHOTCODE':'0x{:04x}'.format
                                           }
        dtypeMapping={}
        for k in ['KSUM00','FSIG00','FSCAT00','NSCALO00','DSGDNPCP','DSGDNSIG',
                  'NSCALO00','MJD-OBS','ZPTMAGAV','SW_PLTSC','FWHM','M5SIGMA','SKYADU']: dtypeMapping[k]=float 
        self.imtable.formattable(dtypeMapping=dtypeMapping,hexcols=['skip'])
        #self.imtable.formattable(dtypeMapping=self.dtypeMapping)

    def format_lctable(self):
        self.default_formatters = {'fluxADU':'{:.2f}'.format, 
                                   'flux_err':'{:.2f}'.format,
                                   'X2norm':'{:.2f}'.format,
                                   'Npix':'{:d}'.format,
                                   'Jyas2':'{:.3e}'.format,     
                                   'Jyas2_err':'{:.3e}'.format,         
                                   'SB':'{:.3f}'.format,  
                                   'SBerr':'{:.3f}'.format,  
                                   'KSUM00':'{:.4f}'.format,
                                   'FSIG00':'{:.3f}'.format,
                                   'FSCAT00':'{:.3f}'.format,
                                   'MJD-OBS':'{:.6f}'.format,
                                   'FWHM':'{:.3f}'.format,
                                   'M5SIGMA':'{:.3f}'.format,
                                   'zpt':'{:.3f}'.format,
                                   'SKYADU':'{:.2f}'.format
                                           }
        dtypeMapping={}
        for k in ['fluxADU','flux_err','Jyas2','Jyas2_err','SB','KSUM00',
                  'FSIG00','FSCAT00','zpt','FWHM','M5SIGMA','SKYADU']: dtypeMapping[k]=float 
        for k in ['ID','x','y','bsize','skip','imID','Npix']: dtypeMapping[k]=int 
        self.formattable(dtypeMapping=dtypeMapping,hexcols=['skip'])
        #self.imtable.formattable(dtypeMapping=self.dtypeMapping)

    def TN(self,filename):
        m = re.search('\.([a-zA-Z0-9]+$)',filename)
        if m is None:
            raise RuntimeError(f'Could not get suffix for {filename}')
        suffix = m.groups()[0] 
        TNname = re.sub('\.[a-zA-Z0-9]+$','.TN.%s' % suffix,filename)
        if TNname == filename:
            raise RuntimeError(f'Could not get thumbnail filename for {filename}')
        return(TNname)


    def init_dirs_and_filenames(self,field,ccd,datarootdir=None,
                                #outputrootdir=None,outsubdir=None,
                                posdir=None,posfilename='auto',check_dirs_exist=True):
        self.field = field
        self.ccd = ccd
        #self.filt = filt
        
        self.fieldccdID = f'{self.field}a{self.ccd}'
        
        # determine the main directories
        if datarootdir is None: datarootdir = '.'
        if posdir is None: posdir = '.'
        #if outputrootdir is None: outputrootdir = '.'
        
        self.datarootdir = os.path.abspath(datarootdir)
        self.tmpl_dir = os.path.join(self.datarootdir,f'tmpl/{self.ccd}')
        self.diff_dir = os.path.join(self.datarootdir,f'{self.field}_tmpl/{self.ccd}')
        print(f'tmpl_dir: {self.tmpl_dir}')
        print(f'diff_dir: {self.diff_dir}')
        if check_dirs_exist:
            if not os.path.isdir(self.tmpl_dir):raise RuntimeError('tmpl directory {self.tmpl_dir} does not exist')
            if not os.path.isdir(self.diff_dir):raise RuntimeError('diffim directory {self.diff_dir} does not exist')

        self.postable.set_posfilename(field,ccd,posdir=posdir,posfilename=posfilename)
                    
    def init_output_dirs_and_filenames(self,outputrootdir=None,outsubdir=None,
                                       filt=None,tmpl_expnum=None):
        """
        This routine sets self.outbasename
        
        It assumes that self.posfilename, self.filt, self.tmpl_expnum, 
        and self.box_size are set to the correct values!

        Parameters
        ----------
        outputrootdir : str, optional
            If set to None, then '.' is used. The default is None.
        outsubdir : str, optional
            If not set to None, outsubdir is added to the output directory. The default is None.

        Returns
        -------
        None.

        """
        # building the output directory and output basenames
        if outputrootdir is None: outputrootdir = '.'
        outputdir = outputrootdir
        if outsubdir is not None: outputdir = os.path.join(outputdir,outsubdir)
        outputdir = os.path.join(outputdir,f'{self.field}/{self.ccd}')
        outputdir = os.path.abspath(outputdir)
        print(f'output directory: {outputdir}')
        
        posfile_basename = re.sub('\.txt$','',os.path.basename(self.postable.posfilename))
        posfile_basename = re.sub('\.lcpos.*','',posfile_basename)
        #posfile_basename = re.sub('\.lcpos\..*','',posfilename)
        
        if filt is None: filt = self.filt
        if tmpl_expnum is None: tmpl_expnum = self.tmpl_expnum
        self.outbasename = os.path.join(outputdir,f'{posfile_basename}_{filt}_tmpl{tmpl_expnum}')
        
        print(f'output basename: {self.outbasename}')
        
        
    def define_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)
        parser.add_argument("field", help="name of the light echo field", type=str)
        parser.add_argument("ccd", help="ddtector number", type=str)
        
        self.define_optional_arguments(parser)
        return(parser)
        
    def define_optional_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        # default directory for input data, like images etc
        if 'ETALC_DATA_ROOTDIR' in os.environ:
            datarootdir = os.environ['ETALC_DATA_ROOTDIR']
        else:
            datarootdir = None

        # default directory for output
        if 'ETALC_OUTPUT_ROOTDIR' in os.environ:
            outputrootdir = os.environ['ETALC_OUTPUT_ROOTDIR']
        else:
            outputrootdir = None

        parser.add_argument("--filters", help="list of filters. If not specified, then all filters from image list are used", type=str)


        parser.add_argument("--datarootdir", default=datarootdir, help=" data root directory  (default=%(default)s)")
        parser.add_argument("--outputrootdir", default=outputrootdir, help=" output root directory  (default=%(default)s)")
        parser.add_argument('--outsubdir', default=None, help='outsubdir added to output root directory (default=%(default)s)')

        parser.add_argument("--box_size", type=int, default=3, help="light curve box size along one side in pixels")
        parser.add_argument("--mask_val", default=0xff00, help="Exclude all pixels with (mask & mask_val)>0 (default=0x%(default)x)")

        parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite files if they exist.')
        parser.add_argument('-s','--save_lcplots', default=False, action='store_true', help='Make light curve plots and save them.')
        parser.add_argument('--show_lcplots', default=False, action='store_true', help='Make light curve plots and show them.')

        parser.add_argument('--skip_createlcs_if_exists', default=False, action='store_true', help='Skip recreating the lightcurves if they already exists. Can be used to just recreate plots etc.')
        parser.add_argument('--skip_createlcplots_if_exists', default=False, action='store_true', help='Skip recreating the lightcurve plots if they already exists.')

        self.postable.define_optional_arguments(parser)

        parser.add_argument('-v','--verbose', default=0, action='count')

        return(parser)
        
    def find_diffims(self,diffim_filepattern=None):
        if diffim_filepattern is None:
            diffim_filepattern = os.path.join(self.diff_dir,f'{self.field}.*_ooi_*.diff.fits')
        if self.verbose: print(f'diffim filepattern: {diffim_filepattern}')
        diff_imgs = sorted(glob.glob(diffim_filepattern))
        #diff_imgs = diff_imgs[:2]
        if len(diff_imgs)==0:
            raise RuntimeError(f'Could not find diffims for pattern {diff_imgs}')
        else:
            print(f'Found {len(diff_imgs)} diff images!')
        
        #diff_imgs = diff_imgs[:2]
        self.imtable.t['diffim']=diff_imgs
        self.imtable.t['imID']=range(len(diff_imgs))
        self.imtable.t['error']=0
        self.imtable.t['skip']=0
        # get teh dcmp filename
        self.imtable.replace_regex('diffim','diffdcmp','\.fits$','.dcmp')
        if self.verbose>1: self.imtable.write()
        
        # Make sure dcmp file exists, if not mark skip=1
        for ix in self.imtable.getindices():
            if not os.path.isfile(self.imtable.t.loc[ix,'diffdcmp']):
                self.imtable.t.loc[ix,'skip']=self.skipmask['missing_dcmp'] # missing_dcmp=0x2
                self.imtable.t.loc[ix,'error']=self.skipmask['missing_dcmp'] # missing_dcmp=0x2
                print(f'WARNING: diffim {self.imtable.t.loc[ix,"diffim"]} does not have an associated dcmp file, SKIPPING!!!!')
        
        ixs_good =  self.imtable.ix_equal('skip',0)
        # get the desired fits header keywords from the diffdcmp file
        self.imtable.fitsheader2table('diffdcmp',indices=ixs_good,
                                      requiredfitskeys=self.fitskeys2imtable_required,
                                      optionalfitskey=self.fitskeys2imtable_optional)
        
        #delme = []
        # Get M5SIGMA, but for this we need to access the sw.dcmp files!
        for ix in self.imtable.getindices():
            swname = f'{self.datarootdir}/{self.imtable.t.loc[ix,"IMSUBD"]}/{self.imtable.t.loc[ix,"IMNAME"]}'
            swdcmp = re.sub('\.fits','.dcmp',swname)
            self.imtable.t.loc[ix,'swdcmp'] = swdcmp
            #delme.append(os.path.basename(swdcmp))
            try: 
                photcode = int(eval(self.imtable.t.loc[ix,'PHOTCODE']))
                self.imtable.t.loc[ix,'PHOTCODE'] = photcode & 0xffff
                self.imtable.t.loc[ix,'filter'] = photcode2filter[photcode & 0xff]
            except:
                self.imtable.t.loc[ix,'PHOTCODE'] = 0
                self.imtable.t.loc[ix,'filter'] = 'x'
                self.imtable.t['error']=1
                self.imtable.t['skip']=1
                
            
            
        #print(' '.join(delme))
        self.imtable.fitsheader2table('swdcmp',indices=ixs_good,
                                      optionalfitskey=['M5SIGMA'])
        
        
        # make sure the format is nice
        self.format_imtable()
        # dcmp filename is not needed anymore
        self.imtable.t.drop(columns=['diffdcmp','swdcmp','IMSUBD','IMNAME'],inplace=True)

        self.imtable.t['tmplexpnum'] = self.imtable.t['diffim'].str.extract(f'{self.field}\..*_ooi_[a-z]+_.*\.(\d+)_ooi_.*.diff.fits')

        if self.verbose: self.imtable.write()

        return(0)

        # This is sanity test: make sure the template expnum is the same than the template expnum in the 
        # difference images, and that there is only one template expnum in the diffims
        tmpl_expnums = self.imtable.t['diffim'].str.extract(f'{self.field}\..*_ooi_{self.filt}_.*\.(\d+)_ooi_.*.diff.fits')
        tmpl_expnums = unique(tmpl_expnums[0].values)
        if len(tmpl_expnums) == 0:
            raise RuntimeError('No template expnums found in the diffims, this is a BUG!!!')
        elif len(tmpl_expnums)>1:
            print(f'template expnums in diffim names: {tmpl_expnums}')
            raise RuntimeError('More than one template expnums found! Not yet implemented....')
        else:
            if self.tmpl_expnum != tmpl_expnums[0]:
                raise RuntimeError(f'template expnum={self.tmpl_expnum} is not equal template expnum={tmpl_expnums[0]} from diffim names')
            # ALL GOOD! only 1 template expnum, and it's the same for diffims and the template
            print(f'All diffims have the template with expnum={self.tmpl_expnum}')

        return(0)
        
    def find_tmpl_delme(self,tmpl_filepattern=None):
        if tmpl_filepattern is None:
            tmpl_filepattern = os.path.join(self.tmpl_dir,f"{self.field}.*_ooi_{self.filt}*.sw.fits")
        if self.verbose: print(f'template filepattern: {tmpl_filepattern}')
        tmpl_names = glob.glob(tmpl_filepattern)
        if len(tmpl_names) == 0:
            raise RuntimeError('No templates found')
        elif len(tmpl_names)>1:
            if self.verbose: print(f'templates: {tmpl_names}')
            raise RuntimeError('More than one template found! Not yet implemented, probably need to add template ID option to give user choices?')
        else:
            self.tmpl_name=tmpl_names[0]
        print(f'template: {self.tmpl_name}')
        m = re.search('\.(\d+)_ooi_',os.path.basename(self.tmpl_name))
        if m is None:
            raise RuntimeError(f'Could not get exnum ID from {os.path.basename(self.tmpl_name)} with pattern "\.(\d+)_ooi_"')
        self.tmpl_expnum = m.groups()[0]
        print(f'template expnum: {self.tmpl_expnum}')
        return(0)
             


    def find_tmpl(self,tmplexpnum):
        tmpl_filepattern = os.path.join(self.tmpl_dir,f"{self.field}.*.{tmplexpnum}_ooi_{self.filt}*.sw.fits")
        if self.verbose: print(f'template filepattern: {tmpl_filepattern}')
        tmpl_names = glob.glob(tmpl_filepattern)
        if len(tmpl_names) == 0:
            raise RuntimeError('No templates found')
        elif len(tmpl_names)>1:
            if self.verbose: print(f'templates: {tmpl_names}')
            raise RuntimeError('More than one template found! Not yet implemented, probably need to add template ID option to give user choices?')
        else:
            self.tmpl_name=tmpl_names[0]
            self.tmpl_expnum=tmplexpnum
        print(f'template: {self.tmpl_name}')
        print(f'template expnum: {self.tmpl_expnum}')
        return(0)
             
    def calc_lcs(self,ixs_diffims=None,box_size=3,mask_val=0xffff):
        
        self.t = pd.DataFrame(columns=self.t.columns)

        
        if self.verbose: print(f'mask_val: 0x{mask_val:x}')
        self.mask_val=mask_val
        
        half_box = int(np.floor(box_size/2))
        # recalculating true box size
        self.box_size = half_box*2+1
                
        ixs_pos = self.postable.getindices()
        
        self.ixs_diffims = self.imtable.getindices(indices=ixs_diffims)
        self.ixs_diffims = self.imtable.ix_sort_by_cols(['MJD-OBS'],indices=ixs_diffims)

        # make a column with noise image
        self.imtable.replace_regex('diffim','diffnoise','\.fits$','.noise.fits',indices=ixs_diffims)
        self.imtable.replace_regex('diffim','diffmask','\.fits$','.mask.fits',indices=ixs_diffims)


        #diff_bases = [diff_img.split('.fits')[0] for diff_img in self.diff_imgs]
        #self.imtable.write()
        #sys.exit(0)
        
        #Open each image and get the flux values for each position
        if self.verbose: print(f'Calculating lc for {len(ixs_diffims)} images')
        for ix_diffim in ixs_diffims:
            diffim_name = self.imtable.t.loc[ix_diffim,'diffim']
            diffnoise_name = self.imtable.t.loc[ix_diffim,'diffnoise']
            diffmask_name = self.imtable.t.loc[ix_diffim,'diffmask']
            
            ### delme just for copying files
            #src_dir1,src_noise = os.path.split(diffnoise_name)
            #src_dir2,src_subdir2 = os.path.split(src_dir1)
            #src_dir3,src_subdir1 = os.path.split(src_dir2)
            #src_noise = f'/astro/armin/data/v20.0/DECAMNOAO/LEec/workspace/{src_subdir1}/{src_subdir2}/{src_noise}'
            #if os.path.isfile(diffnoise_name):
            #    print(f'skipping {diffnoise_name}')
            #    continue
            #print(f'cp {src_noise} {diffnoise_name}')
            #src_mask = f'/astro/armin/data/v20.0/DECAMNOAO/LEec/workspace/{src_subdir1}/{src_subdir2}/{os.path.basename(diffmask_name)}'
            #if os.path.isfile(diffmask_name):
            #    print(f'skipping {diffmask_name}')
            #    continue
            #print(f'cp {src_mask}  {diffmask_name}')
            #continue
            
            if (self.imtable.t.loc[ix_diffim,'DIFFSUCC']==0):
                # this should VERY rarely happen!!
                print(f'##########\n##########  WARNING!! diffim {diffim_name} has DIFFSUCC=0 on fits header, skipping!')
                self.imtable.t.loc[ix_diffim,'skip']=self.skipmask['pipe_nosuccess'] # pipe_nosuccess=0x4
                self.imtable.t.loc[ix_diffim,'error']=self.skipmask['pipe_nosuccess'] # pipe_nosuccess=0x4
                continue

            try:
                #get the image, mask, and noise data
                with fits.open(diffim_name) as diffim_hdu:
                    img = diffim_hdu[0].data
                with fits.open(diffnoise_name) as diffnoise_hdu:
                    nse = diffnoise_hdu[0].data
                with fits.open(diffmask_name) as diffmask_hdu:
                    msk = diffmask_hdu[0].data
            except:
                print(f'##########\n##########  WARNING!! one of the files for {diffim_name} does not exist, skipping!')
                self.imtable.t.loc[ix_diffim,'skip']=self.skipmask['missing_noise'] # missing_noise=0x2
                self.imtable.t.loc[ix_diffim,'error']=self.skipmask['missing_noise'] # missing_noise=0x2
                continue

            if self.verbose>1: print(f'Getting fluxes for {diffim_name}')
            
            #mjd = float(hdr['MJD-OBS'])
            #zpmag = float(hdr['ZPTMAGAV'])
            #pixscale = float(hdr['SW_PLTSC']) #arcsec per pix
            
            mjd = self.imtable.t.loc[ix_diffim,'MJD-OBS']
            if self.imtable.t.loc[ix_diffim,'PHOTNORM'] == 't':
                zpmag = self.imtable.t.loc[ix_diffim,'TZPTMAG']
                if self.verbose>1: print(f'Using template zeropoint {zpmag}')
            else:    
                zpmag = self.imtable.t.loc[ix_diffim,'ZPTMAGAV']
                if self.verbose>1: print(f'Using ZPTMAGAV={zpmag}')
            pixscale = self.imtable.t.loc[ix_diffim,'SW_PLTSC'] #arcsec per pix
            
            for ix_pos in ixs_pos:
                xpix = self.postable.t.loc[ix_pos,'x']
                ypix = self.postable.t.loc[ix_pos,'y']
                
                # get the pixel values for image, noise, and mask
                img_box = img[ypix-half_box:ypix+half_box+1,xpix-half_box:xpix+half_box+1]
                nse_box = nse[ypix-half_box:ypix+half_box+1,xpix-half_box:xpix+half_box+1]
                mask_box = msk[ypix-half_box:ypix+half_box+1,xpix-half_box:xpix+half_box+1]
                
                
                # only use unmasked image, based on mask_val
                unmasked_pixels = np.where(bitmask.bitfield_to_boolean_mask(mask_box,ignore_flags=~mask_val,good_mask_value=True))
                img_box_unmasked = img_box[unmasked_pixels]
                nse_box_unmasked = nse_box[unmasked_pixels]
                wht_box_unmasked = 1.0/(np.square(nse_box_unmasked))
                    
                ix_lc = self.newrow({'ID':self.postable.t.loc[ix_pos,'ID'],
                                        'x':self.postable.t.loc[ix_pos,'x'],
                                        'y':self.postable.t.loc[ix_pos,'y'],
                                        'bsize':self.box_size,
                                        'mjd':mjd,
                                        'zpt':zpmag,
                                        'fluxADU':np.nan,
                                        'flux_err':np.nan,
                                        'X2norm':np.nan,
                                        'Npix':0,
                                        'Nmask':0,
                                        'Jyas2':np.nan,
                                        'Jyas2_err':np.nan,
                                        'SB':np.nan,
                                        'SB_err':np.nan
                                        })
                if np.sum(wht_box_unmasked) == 0.0:
                    if self.verbose>1: print(f'Warning: no flux at {self.postable.t.loc[ix_pos,"x"]} {self.postable.t.loc[ix_pos,"y"]} for image {diffim_name}')
                else:
                    fluxADU = np.average(img_box_unmasked, weights = wht_box_unmasked)  #ADU per pixel
                    errADU = np.sqrt(1.0/np.sum(wht_box_unmasked))
                    if len(img_box_unmasked)>1:
                        res = img_box_unmasked - np.full(img_box_unmasked.shape, fluxADU)
                        chi_norm = 1.0/(len(img_box_unmasked)-1) * np.sum(np.square(np.divide(res,nse_box_unmasked.flatten())))
                    else:
                        chi_norm = np.nan
                        
                        
                    #print('ffff',fluxADU,errADU)
    
                    Jyas2 = 3631.0 * 10**(-0.4*zpmag) * fluxADU / (pixscale**2)
                    Jyas2_err = 3631.0 * 10**(-0.4*zpmag) * errADU / (pixscale**2)
                    
                    self.t.loc[ix_lc,'fluxADU']=fluxADU
                    self.t.loc[ix_lc,'flux_err']=errADU
                    self.t.loc[ix_lc,'X2norm']=chi_norm
                    self.t.loc[ix_lc,'Npix']=len(img_box_unmasked)
                    self.t.loc[ix_lc,'Nmask']=len(img_box[np.where(bitmask.bitfield_to_boolean_mask(mask_box,ignore_flags=~0xffff,good_mask_value=True))])
                    self.t.loc[ix_lc,'Jyas2']=Jyas2
                    self.t.loc[ix_lc,'Jyas2_err']=Jyas2_err
    
                    if fluxADU/errADU>=3.0:
                        fluxas2 =  fluxADU / (pixscale**2)
                        erras2 =  errADU / (pixscale**2)
        
                        self.t.loc[ix_lc,'SB']=-2.5*np.log10(fluxas2)+zpmag
                        self.t.loc[ix_lc,'SB_err'] = 2.5 / np.log(10.0) * erras2/fluxas2
                #for col in self.cols_im2lc:
                self.t.loc[ix_lc,'skip']= int(self.imtable.t.loc[ix_diffim,'skip'])
                self.t.loc[ix_lc,self.cols_im2lc] = self.imtable.t.loc[ix_diffim,self.cols_im2lc]

            #self.format_lctable()
            #self.write()
            #sys.exit(0)
        
        #self.formattable(hexcols=['skip'])
        self.format_lctable()
        self.imtable.t.drop(columns=['diffnoise','diffmask'],inplace=True)

    def load_lcfiles(self,outbasename=None):
        if outbasename is None:
            outbasename = self.outbasename

        lcfile = f'{outbasename}_lc.txt'
        self.load(lcfile)
        self.format_lctable()

        xyposfilename = f'{outbasename}_lcpos.xy.txt'
        self.postable.load(xyposfilename,verbose=2)

        imtablefilename = f'{outbasename}_diffims.txt'
        self.imtable.load(imtablefilename)
        self.format_imtable()
        
        return(0)
        
        
    def save_lcfiles(self,outbasename=None):
        if outbasename is None:
            outbasename = self.outbasename
        
        lcfile = f'{outbasename}_lc.txt'
        self.format_lctable()
        ixs_lc = self.ix_sort_by_cols(['ID','mjd'])
        self.write(lcfile,indices=ixs_lc,overwrite=True,verbose=2)
        
        xyposfilename = f'{outbasename}_lcpos.xy.txt'
        self.postable.write(xyposfilename,verbose=2)

        imtablefilename = f'{outbasename}_diffims.txt'
        self.format_imtable()
        self.imtable.write(imtablefilename,verbose=2,indices=self.ixs_diffims)
        
        print('Saving individual lc files for each ID')
        for ix_pos in self.postable.getindices():
            ID = self.postable.t.loc[ix_pos,'ID']
            ixs_lc_ID = self.ix_equal('ID',ID)
            ixs_lc_ID = self.ix_sort_by_cols(['mjd'],indices=ixs_lc_ID)
            outfile = f'{outbasename}_ID{ID}_lc.txt'
            #print(f'Saving {outfile}')
            self.write(outfile,overwrite=True,indices=ixs_lc_ID,verbose=self.verbose)

    def clean_lc(self,maxerr=7e-07,indices=None):
        ixs = self.ix_not_null(['mjd','Jyas2','Jyas2_err'],indices=indices)

        Nall=len(ixs)
        ixs = self.ix_inrange('Jyas2_err',None,maxerr,indices=ixs)
        print(f'REmoving {Nall-len(ixs)} out of {Nall} because uncertainties are too large')

        return(ixs)
            
    def plotlelc(self,ixs,sp=None,fig=None,fmt='bo',mjdcol='mjd',fluxcol='Jyas2',fluxerrcol='Jyas2_err',
                 xlim=None,ylim=None,ms=None,
                 add2y=None,add2x=None,multy=None,yoffset=None,
                 savefig=None,makeTN=False,thumbnailscale=7.0):
        if sp is None or fig is None:
            (sps,fig)=initplot(1,1)
            sp = sps[0]

        ixs = self.ix_not_null([mjdcol,fluxcol,fluxerrcol],indices=ixs)
                   
        mjd = self.t.loc[ixs,mjdcol].values
        flux = self.t.loc[ixs,fluxcol].values
        fluxerr = self.t.loc[ixs,fluxerrcol].values
        
        flux=flux*1E6
        fluxerr=fluxerr*1E6
        ylabel='$\mu$Jy'
        
        if add2y is not None:
            flux += add2y
        if add2x is not None:
            mjd += add2x
        if multy is not None:
            flux *= multy
            fluxerr *= multy
        if yoffset is not None:
            flux += yoffset
            sp.axhline(yoffset,  color='black',linestyle='--', linewidth=0.5)
        
        sp.axhline(0.0,  color='black',linestyle='--', linewidth=1.0)

        sp, plot, dplot = plotdata(mjd, flux, dy=fluxerr, fmt=fmt)
        sp.set_xlabel('MJD',fontsize=20)
        sp.set_ylabel(f'difference image flux ({ylabel})',fontsize=20)
        plt.setp(sp.get_xticklabels(),'fontsize',14)
        plt.setp(sp.get_yticklabels(),'fontsize',14)
                        
        if ms is not None:    
            plt.setp(plot,'ms',float(ms))
        
        if ylim is not None:
            sp.set_ylim(ylim)

        
        #self.t.loc[ixs].plot(kind='scatter',
        #                   x=mjdcol,y=fluxcol, yerr=fluxerrcol, 
        #                   ax=sp,ylim=ylim, xlim=xlim, legend = False, 
                          # **plot_style['good_data']
        #                  )
        #self.t.loc[ixs].plot(mjdcol,fluxcol,yerr=fluxerrcol,
        #                   ax=sp,color='black')
        
        plt.tight_layout()
        
        if savefig is not None:
            if savefig.lower()=='auto':
                outfilename = f'{self.filename}.png'
            else:
                outfilename=savefig
            if self.verbose: print(f'Saving plot to {outfilename}')
            plt.savefig(outfilename)
            if makeTN:
                figsize = fig.get_size_inches()
                figsize *= 1.0/thumbnailscale
                fig.set_size_inches(figsize[0], figsize[1] )
                #Size = fig.get_size_inches()
                #print "Size in Inches", Size
                if self.verbose>1:print("Saving", self.TN(outfilename))
                fig.savefig(self.TN(outfilename))

            
            
        return(sp,fig)

                
    def mk_lcplots(self,IDs=None,xypos_indices=None,outbasename=None,lc_indices=None,
                   save_lcplots=False,show_lcplots=False,
                   skip_createlcplots_if_exists=False,
                   makeTN=False):
        if outbasename is None:
            outbasename = self.outbasename

        lc_ixs = self.getindices(lc_indices)

        if IDs is None and xypos_indices is None:
            IDs = unique(self.t.loc[lc_ixs,'ID'].values)
        else:
            if xypos_indices is not None:
                IDs = unique(self.postable.t.loc[xypos_indices,'ID'].values)
        IDs=sorted(IDs)
        
        print('making individual lc plots for each ID')
        for ID in IDs:
            ixs_lc_ID = self.ix_equal('ID',ID)
            ixs_lc_ID = self.ix_sort_by_cols(['mjd'],indices=ixs_lc_ID)
            if save_lcplots:
                outfile = f'{outbasename}_ID{ID}_lc.png'
            else:
                outfile = None
            if skip_createlcplots_if_exists and os.path.isfile(outfile):
                if self.verbose: print(f'skipping recreating {outfile}')
                continue
            if self.verbose>1:  print(f'Making plots for postion ID={ID}')
            (sp,fig) = self.plotlelc(ixs_lc_ID,savefig=outfile,makeTN=makeTN)
            if show_lcplots:
                plt.show()
            plt.close(fig)

    def mk_main_webpage(self,outbasename=None,htmlname=None):
        if outbasename is None:
            outbasename = self.outbasename
        if htmlname is None:
            htmlname = f'{os.path.dirname(outbasename)}/index.html'
        
        for ix in self.webpagelist.getindices():
            self.webpagelist.t.loc[ix,'link'] = addlink2string(os.path.basename(self.webpagelist.t.loc[ix,'html']),os.path.basename(self.webpagelist.t.loc[ix,'html']))
            
        # write the table to index.html
        print(f'writing html to {htmlname}')
        f=open(htmlname,'w')
        #s_asciilink = addlink2string('ascii-table',os.path.basename(asciiname))
        #f.writelines([f'Level 1+2 Products for {description} ({s_asciilink} here)'])
        (errorflag,lines)=self.webpagelist.write(return_lines=True, columns=['fieldccdID','filter','tmpl_expnum','link'], 
                                                 htmlflag=True, htmlsortedtable=True, escape=False)
        if errorflag: raise RuntimeError(f'Soemthing went wrong when doing the webpage table for {outbasename}')
        f.writelines(lines)
        f.close()


    def mk_webpage_filter_expnum(self,xypos_indices=None,outbasename=None,
                   p='100%',
                   htmlname=None):
        if outbasename is None:
            outbasename = self.outbasename
        if htmlname is None:
            htmlname = f'{os.path.dirname(outbasename)}/index.html'

        xypos_indices = self.postable.getindices(xypos_indices)
        
        postable = copy.deepcopy(self.postable)
        
        postable.t['LC']=None
        for ix_xypos in xypos_indices:
            ID = postable.t.loc[ix_xypos,'ID']
            plotfilename = f'{outbasename}_ID{ID}_lc.png'
            plotTNfilename = self.TN(plotfilename)
            
            lcfilename = f'{outbasename}_ID{ID}_lc.txt'
            postable.t.loc[ix_xypos,'ID'] = addlink2string(postable.t.loc[ix_xypos,'ID'],os.path.basename(lcfilename))
            
            postable.t.loc[ix_xypos,'LC']=addlink2string(imagestring4web(os.path.basename(plotTNfilename),width=None,height=p),os.path.basename(plotfilename))

            
        # write the table to index.html
        print(f'writing html to {htmlname}')
        f=open(htmlname,'w')
        #s_asciilink = addlink2string('ascii-table',os.path.basename(asciiname))
        #f.writelines([f'Level 1+2 Products for {description} ({s_asciilink} here)'])
        (errorflag,lines)=postable.write(return_lines=True, indices=xypos_indices,  
                                         htmlflag=True, htmlsortedtable=True, escape=False)
        if errorflag: raise RuntimeError(f'Soemthing went wrong when doing the webpage table for {outbasename}')
        f.writelines(lines)
        f.close()
        
        # Make sure sortable.js is in the html directory
        htmldir = os.path.dirname(htmlname)
        if not ("ETALC_ROOTDIR" in os.environ):
            raise RuntimeError("environment variable ETALC_ROOTDIR does not exist!")
        jsfilename = f'{os.environ["ETALC_ROOTDIR"]}/sortable.js'
        dest_jsfilename = f'{htmldir}/sortable.js'
        if not os.path.isfile(dest_jsfilename):
            if not os.path.isfile(jsfilename):
                raise RuntimeError(f'java script {jsfilename} for sortable tables does not exist!')
            shutil.copy(jsfilename, dest_jsfilename)

        self.webpagelist.newrow({'fieldccdID':self.fieldccdID,
                                 'filter':self.filt,
                                 'tmpl_expnum':self.tmpl_expnum,
                                 'html':htmlname})

        """
        for ID in IDs:
            ixs_lc_ID = self.ix_equal('ID',ID)
            ixs_lc_ID = self.ix_sort_by_cols(['mjd'],indices=ixs_lc_ID)
            if savefigflag:
                outfile = f'{outbasename}_ID{ID}_lc.png'
            else:
                outfile = None
            if skip_createlcplots_if_exists and os.path.isfile(outfile):
                if self.verbose: print(f'skipping recreating {outfile}')
                continue
            print(f'making plots for postion ID={ID}')
        """
            
if __name__=='__main__':
    etalc = extract_LE_lcclass()
    parser = etalc.define_arguments()
    args = parser.parse_args()
    
    if isinstance(args.mask_val,str):
        mask_val = int(eval(args.mask_val))
    else:
        mask_val = args.mask_val
    
    etalc.verbose=args.verbose
    
    etalc.init_dirs_and_filenames(args.field,args.ccd,
                                  datarootdir=args.datarootdir,posdir=args.posdir,
                                  posfilename=args.posfilename)

    # find images
    etalc.find_diffims()
    filters = sorted(unique(etalc.imtable.t['filter']))
    if args.filters is not None:
        filters=AandB(filters,args.filters)
    print(f'Filters: {" ".join(filters)}')

    # Load the positions
    etalc.postable.load_posfile(imagename=etalc.imtable.t.loc[0,'diffim'])
    
    for filt in filters:
        etalc.filt=filt
        ixs_filter = etalc.imtable.ix_equal('filter',filt)
        tmplexpnums = sorted(unique(etalc.imtable.t.loc[ixs_filter,'tmplexpnum']))
        print(f'\n#########################\n### Filter {filt}, template expnums: {" ".join(tmplexpnums)}\n#########################')
        
        for tmplexpnum in tmplexpnums:
            ixs_tmplexpnum = etalc.imtable.ix_equal('tmplexpnum',tmplexpnum,indices=ixs_filter)
            print(f'### template expnum {tmplexpnum}: {len(ixs_tmplexpnum)} images')
            etalc.find_tmpl(tmplexpnum)
            
            # calculate the lcs
            if args.skip_createlcs_if_exists:
                # recalculating true box size
                print('!!! skipping recreating lcs!!!! This is mainly to debug the webpages...')
                etalc.ixs_diffims = ixs_tmplexpnum
                etalc.box_size = int(np.floor(args.box_size/2))*2+1
                etalc.mask_val = mask_val
                etalc.init_output_dirs_and_filenames(outputrootdir=args.outputrootdir,outsubdir=args.outsubdir)
                etalc.load_lcfiles()
                #etalc.imtable.write()
            else:
                etalc.calc_lcs(ixs_diffims=ixs_tmplexpnum,box_size=args.box_size, mask_val = mask_val)
                etalc.init_output_dirs_and_filenames(outputrootdir=args.outputrootdir,outsubdir=args.outsubdir)
                etalc.save_lcfiles()

            if not args.save_lcplots:
                print('\n!!!! Skipping making plots!! If you want to recreate plots, use --save_lcplots or -s!!!!\n')
            else:
                etalc.mk_lcplots(save_lcplots=True,show_lcplots=args.show_lcplots,
                                 skip_createlcplots_if_exists=args.skip_createlcplots_if_exists,
                                 makeTN=True)

            
            if etalc.verbose>1: etalc.write()
          
        
    sys.exit(0)
    etalc.find_tmpl()
    # Load the positions
    etalc.postable.load_posfile(imagename=etalc.tmpl_name)
    
    # calculate the lcs
    if args.skip_createlcs_if_exists:
        # recalculating true box size
        print('!!! skipping recreating lcs!!!!')
        etalc.box_size = int(np.floor(args.box_size/2))*2+1
        etalc.mask_val = mask_val
        etalc.init_output_dirs_and_filenames(outputrootdir=args.outputrootdir,outsubdir=args.outsubdir)
        etalc.load_lcfiles()
        #etalc.imtable.write()
    else:
        etalc.calc_lcs(box_size=args.box_size, mask_val = mask_val)
        etalc.init_output_dirs_and_filenames(outputrootdir=args.outputrootdir,outsubdir=args.outsubdir)
        etalc.save_lcfiles()
    if etalc.verbose>1: etalc.write()
    etalc.mk_lcplots(save_lcplots=args.save_lcplots,show_lcplots=args.show_lcplots,
                     skip_createlcplots_if_exists=args.skip_createlcplots_if_exists,
                     makeTN=True)
    etalc.mk_webpage_filter_expnum()