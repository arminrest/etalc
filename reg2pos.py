#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:41:35 2023

@author: arest
"""

import os,sys,re,copy,shutil
import numpy as np
import pandas as pd
import glob
import argparse

from extract_ETA_lcs import positionclass
from pdastro import pdastroclass,unique


class reg2posclass(positionclass):
    """
    class for the table of positions for the lc's
    """
    def __init__(self,verbose=0):
        positionclass.__init__(self)
        self.verbose=0
        
        self.radec_searchpattern = 'circle\(([0-9\:\.]+)\,([0-9\:\-\.]+)|([0-9\:\.]+)\,([0-9\:\-\.]+)'
        
    def define_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)
        parser.add_argument("regionfiles", nargs="+", help="list of names of the ds9 region file", type=str)
        parser.add_argument("-o","--outputfile", default='auto', help="output position filename. if 'auto', then the output filename is equal the region filename, with .reg substituted with lcpos.txt", type=str)
        parser.add_argument("--globalcolor", default=None, help="define the global color. If None, then script will try to glean it from region file  (default=%(default)s)", type=str)
        parser.add_argument("-a","--add2file", action='store_true', default=None, help='Add the new positions to the output file specified with --outputfile. This does not work with outputfile="auto".')
        parser.add_argument('-v','--verbose', default=0, action='count')
        
        self.define_optional_arguments(parser)
        return(parser)
    
    def convert_region2pos(self,regioninfo,globalcolor=None,group0=None,ID0=None):
        """

        Parameters
        ----------
        regioninfo : String or list/tuple
            If string, then regioninfo is assumed to be teh region filename
            If list/tuple, then this is the list with the region info
        globalcolor : string, optional
            globalcolor is used when no color is specified in a region line. 
            globalcolor gets overwrittin if there is a line "global ... color=X ..." in the region info
        group0: int, optional
            start ID. If None, then determined automatically
        ID0: int, optional
            start ID. If None, then determined automatically
            
        Raises
        ------
        RuntimeError
            Error gets raised if there is not "fk5" in the region info.
            This is to make sure that we assume the correct coordinate system

        Returns
        -------
        None.

        """
        if isinstance(regioninfo,str):
            # is region info is string, we assume it is a filename
            if self.verbose>1: print(f'### Loading {regioninfo}')
            lines=open(regioninfo).readlines()
        elif isinstance(regioninfo,(list,tuple)):
            lines=regioninfo
        else:
            raise RuntimeError(f'Can\'t figure out what to do with region info {regioninfo}')
        fk5=False
        
        pattern=re.compile(self.radec_searchpattern)
        
        tmptable = pdastroclass()

        for line in lines:
            line=line.strip()
            if self.verbose>2: print(f'{line}')
            if re.search('^global',line) is not None:
                colorinfo=re.search('color\=(\w+)',line)
                if colorinfo is not None:
                    globalcolor=colorinfo.groups()[0]
                if self.verbose>1: print('global color:',globalcolor)
            if re.search('^fk5',line) is not None:
                fk5=True
                if self.verbose>1:  print('fk5 confirmed!')
            radec = pattern.search(line)
            if radec is not None:
                ra=radec.groups()[0]
                dec=radec.groups()[1]
                if ra is None:
                    ra=radec.groups()[2]
                    dec=radec.groups()[3]
                    if ra is None:
                        raise RuntimeError('BUG???')
                colorinfo=re.search('color\=(\w+)',line)
                if colorinfo is not None:
                    color=colorinfo.groups()[0]
                else:
                    color=globalcolor
                
                tmptable.newrow({'group':None,
                                 'ID':None,
                                 'ra':ra,
                                 'dec':dec,
                                 'color':color})

        if not fk5:
            raise RuntimeError('Could not determine if coordinates are in fk5!')
        
        
        # Get the starting group and ID. Depends on what is passed in the option, 
        # and if there are already entries!
        if len(self.t)==0:
            if group0==None: group0=0
            if ID0==None: ID0=0
        else:
            if group0==None:
                group0=self.t['group'].max()+1
            if ID0==None:
                ID0=self.t['ID'].max()+1
        
        colors = unique(tmptable.t['color'])
        for color in colors:
            # Make the first index of a group always a multiple of 10            
            if ID0 % 10 !=0:
                ID0 = ID0 - (ID0 % 10) + 10

            ixs_color = tmptable.ix_equal('color',color)
            tmptable.t.loc[ixs_color,'group']=group0
            tmptable.t.loc[ixs_color,'ID']=range(ID0,ID0+len(ixs_color))
            
            # increment the group and ID!
            ID0+=len(ixs_color)
            group0 += 1
        
        if self.verbose>1:
            ixs = tmptable.ix_sort_by_cols(['group','ID'])
            tmptable.write(indices=ixs)
           
        tables=[]
        if len(self.t)>0:
            tables.append(self.t)
        tables.append(tmptable.t)
        self.t = pd.concat(tables, axis=0, ignore_index=True)
        if self.verbose>1:
            print(f'{len(tmptable.t)} entries added (groups {unique(tmptable.t["group"])}), total {len(self.t)}')
        

if __name__=='__main__':
    reg2pos = reg2posclass()
    parser = reg2pos.define_arguments()
    args = parser.parse_args()
    
    reg2pos.verbose=args.verbose
    
    if args.add2file:
        if args.outputfile == 'auto': raise RuntimeError("outputfile='auto' not allowed with add2file flag!")
        reg2pos.load_posfile(posfilename=args.outputfile,checkxyflag=False)
    if args.outputfile == 'auto' and len(args.regionfiles)>1:
        raise RuntimeError("outputfile='auto' not allowed with more than 1 input region file!")

    
    for regionfile in args.regionfiles:
        reg2pos.convert_region2pos(regionfile,globalcolor=args.globalcolor)
        
    if args.outputfile == 'auto': 
        outputfilename = re.sub('\.reg','.lcpos.txt',args.regionfiles[0])
        if outputfilename==args.regionfiles[0]: raise RuntimeError(f'Could not derive auto outputfilename from {args.regionfiles[0]}')
    else:
        outputfilename = args.outputfile 
    ixs = reg2pos.ix_sort_by_cols(['group','ID'])
    if reg2pos.verbose:
        reg2pos.write(indices=ixs)
    reg2pos.write(outputfilename,indices=ixs,verbose=2)
           