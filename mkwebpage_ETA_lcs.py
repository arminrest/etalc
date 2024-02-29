#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:03:22 2024

@author: arest
"""

import os,sys,re,copy,shutil
import glob
import argparse

from pdastro import AandB,unique
from extract_ETA_lcs import extract_LE_lcclass

def get_tmpl_expnums(filepattern,suffix='_lcpos.xy.txt'):
    print(f'Looking for files with filepattern {filepattern}')
    files = glob.glob(f'{filepattern}*{suffix}')
    tmpl_expnums = []
    s = f'_tmpl(\d+){suffix}'
    expnumpattern = re.compile(s)
    for filename in files:
        m = expnumpattern.search(filename)
        if m is None:
            raise RuntimeError(f'BUG??? could not find pattern {s} in {filename}')
        tmpl_expnums.append(int(m.groups()[0]))
    tmpl_expnums = sorted(unique(tmpl_expnums))
    return(tmpl_expnums)
    
if __name__=='__main__':
    etalc = extract_LE_lcclass()
    parser = etalc.define_arguments()
    #parser.add_argument('--makefigs', default=False, action='store_true', help='Make the light curve plots.')

    args = parser.parse_args()
        
    etalc.verbose=args.verbose
    
    etalc.init_dirs_and_filenames(args.field,args.ccd,
                                  datarootdir=args.datarootdir,posdir=args.posdir,
                                  posfilename=args.posfilename,check_dirs_exist=False)

    filters = ['i','g','r','z']
    if args.filters is not None:
        filters=AandB(filters,args.filters)

    for filt in filters:
        print(f'\n### filter {filt}')
        etalc.filt=filt
        
        # preliminary initialization of the outpdirs in order to get the basename for the file
        etalc.init_output_dirs_and_filenames(outputrootdir=args.outputrootdir,outsubdir=args.outsubdir,
                                             tmpl_expnum='')
        # get the template expnums for the filter
        tmpl_expnums = get_tmpl_expnums(etalc.outbasename)
        print(f'template expnums: {tmpl_expnums}')
        for tmpl_expnum in tmpl_expnums:
            print(f'## Making webpage for template expnum {tmpl_expnum}')
            etalc.tmpl_expnum = tmpl_expnum
            etalc.init_output_dirs_and_filenames(outputrootdir=args.outputrootdir,outsubdir=args.outsubdir)
            etalc.load_lcfiles()
            if not args.save_lcplots:
                print('\n!!!! Skipping making plots!! If you want to recreate plots, use --save_lcplots or -s!!!!\n')
            else:
                etalc.mk_lcplots(save_lcplots=True,show_lcplots=args.show_lcplots,
                                 skip_createlcplots_if_exists=args.skip_createlcplots_if_exists,
                                 makeTN=True)
            htmlname=f'{etalc.outbasename}.html'
            etalc.mk_webpage_filter_expnum(htmlname=htmlname)
    etalc.webpagelist.write()
    etalc.mk_main_webpage()
