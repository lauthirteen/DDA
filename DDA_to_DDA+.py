#!/usr/bin/python3
# -*- coding: utf-8 -*
'''
Switch surfactant conversion:
  The conversion rate can be set to simulate
the transition of the monomolecular structure
of the surfactant after it is injected with CO2.

Gromacs version: 2023.2.
  You can modify the 'RunGmx' function to make this
code applicable to the other GMX version


            By Jianchuan Liu  XHU  2025-05-13
'''

import linecache
import sys
import math
import numpy as np
import re
import time
import datetime
import re
import collections
from itertools import combinations
from itertools import product
import itertools
import random
from shutil import copyfile
import glob
import os
import shutil
#####################################################################################
def ReadGMXGro(GroFile):
    '''
    Read single frame .gro file,
    :param GroFile:
    :return: box, moles. moles: list consisted of Molecule class.
    '''
    with open(GroFile, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    totalmoles = []
    openedfile = open(GroFile,'r')
    natoms = len(openedfile.readlines()) - 3
    boxv = np.array(list(map(float, lines[2 + natoms].split())))

    atoms = lines[2: 3 + natoms]
    n = 0
    while n < natoms:
        name = atoms[n][:10].strip()
        typename = atoms[n][5:10].strip()
        coordinates = []
        symbols = []
        mole = []
        for nextn in range(n, natoms + 1):
            coordinates.append(list(map(float, atoms[nextn][20:44].split())))
            symbols.append((atoms[nextn][10:15]).strip())
            if nextn + 1 == natoms or atoms[nextn + 1][:10].strip() != name:
                break
        line1 = 0
        for info1 in symbols:
            atomlist = []
            x = coordinates[line1][0]
            y = coordinates[line1][1]
            z = coordinates[line1][2]
            atomlist.append(typename)
            atomlist.append(info1)
            atomlist.append(x)
            atomlist.append(y)
            atomlist.append(z)
            mole.append(atomlist)
            line1 += 1
        totalmoles.append(mole)
        n = nextn + 1
    openedfile.close()
    f.close()
    return natoms, boxv, totalmoles

def GetComCoord(atom_list):
    '''
    Read single molecule coordnates list,
    :param atom_list: single molecule coordnates list
    :return:  com coordnates list (x, y, z)
    '''
    # atomic mass
    mass_dict = {'C':12.010,'H':1.000,'N':14.010,'F':0.000,'S':32.060,'O':16.000,'M':0.000, 'X': 28.000, 'P': 24.02}
    mass_sum = 0
    x_mass_sum = 0
    y_mass_sum = 0
    z_mass_sum = 0

    com_coord = []
    for info in atom_list:
        x = info[-3]
        y = info[-2]
        z = info[-1]
        atom = info[-4][0:1]
        atom_mass = mass_dict[atom]
        mass_sum += float(atom_mass)
        x_mass_sum += float(x) * float(atom_mass)
        y_mass_sum += float(y) * float(atom_mass)
        z_mass_sum += float(z) * float(atom_mass)

    x_com = round(x_mass_sum / mass_sum,3)
    y_com = round(y_mass_sum / mass_sum,3)
    z_com = round(z_mass_sum / mass_sum,3)

    com_coord.append(x_com)
    com_coord.append(y_com)
    com_coord.append(z_com)

    return com_coord

def GetComDist(coord1, coord2):
    dist = round(math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2), 3)
    return dist

def CreateNewGro(new_gro_file,new_gro_list, resid, natom):
    '''
    Read new gro file name, new gro list and box vector
    :param new_gro_file:
    :param new_gro_list:
    :param resid:
    :param natom:
    :return:
    '''
    all_resname = []
    #read total molecle
    if len(new_gro_list) > 0:
        for info1 in new_gro_list:
            all_resname.append(info1[0][0])
            # read single molecle
            for info2 in info1:
                # read single atomic information
                resname = info2[0]
                atomtype = info2[1]
                x = info2[2]
                y = info2[3]
                z = info2[4]
                natom = (natom - 1) % 99999 + 1
                resid = (resid - 1) % 99999 + 1
                new_gro_file.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" %
                              (resid, resname, atomtype, natom, x, y, z))
                natom += 1
            resid += 1
    else:
        resid = resid
        natom = natom
    return all_resname, resid, natom

def RunGmx(run_init, em_mdp_name, md_mdp_name,version=2023):
    '''
    Run the Gromacs program
    :param run_init: the number of run time
    :param em_mdp_name: the name of energy minimization MDP of gmx
    :param md_mdp_name: the name of energy minimization MDP of gmx
    :return: none
    '''
    if version == 4:
        version1 = ''
    else:
        version1 = 'gmx'
    os.system('%s grompp -f %s -c step_%s.gro -p step_%s.top -o em_%s.tpr -maxwarn 7 >& /dev/null'
              % (version1, em_mdp_name,run_init,run_init,run_init))
    os.system('%s mdrun -v -deffnm em_%s -nt 10 > em_out.log 2>&1' % (version1, run_init))
    os.system('%s grompp -f %s -c em_%s.gro -p step_%s.top -o md_%s.tpr -maxwarn 7 >& /dev/null'
              % (version1, md_mdp_name,run_init,run_init,run_init))
    os.system('%s mdrun -v -deffnm md_%s -nt 10 > md_out.log 2>&1' % (version1, run_init))

def TranslationCoordinates(target_molecule_list, target_com_coord, substituted_com_coord, x, y, z):
    '''
    It shifts the molecule to a particular position
    :param target_molecule_list: The coordinates of the target molecule
    :param substituted_molecule_list: The coordinates of the substituted molecule
    :return: The coordinates of the molecules after translation
    '''
    vector_list = []
    for i, j in zip(substituted_com_coord, target_com_coord):
        subtr = i - j
        vector_list.append(subtr)
    new_coord = []
    for i, j in zip(vector_list, target_molecule_list):
        summ = i + j
        new_coord.append(round(summ, 3))
    # The center of mass migration
    new_coord[0] = new_coord[0] + x
    new_coord[1] = new_coord[1] + y
    new_coord[2] = new_coord[2] + z
    return new_coord

def GetLineNum(FileName, KeyWord):
    '''
    fine line number
    :param FileName:
    :param KeyWord:
    :return: LineList list and LineTotal
    '''
    filename = open(FileName,'r')
    LineTotal = len(filename.readlines())
    n = 0
    LineList = []
    while n <= LineTotal:
        LineData = str(linecache.getline(FileName, n))
        #info = LineData.find(KeyWord)
        info = re.search(KeyWord, LineData)
        if info == None:
            pass
        else:
            LineList.append(n)
        n += 1
    filename.close()
    return LineList, LineTotal

def WroInfo(LineNum):
    '''
    print wrong information
    '''
    print("!"*46)
    print("!!!!! The Gromacs program running error  !!!!!")
    print("!!!!!  The program has been terminated   !!!!!")
    print("!!!!!    All output files are invalid    !!!!!")
    print("!!!!! Please check the code, line %s    !!!!!" % LineNum)
    print("!"*46)
    sys.exit()
    return

def GmxJudge(CodeLineNum,name):
    '''
    Judege Whether to execute successfully for Gmx
    '''
    if (os.path.exists(name) != True) or (os.path.getsize(name) == 0):
        WroInfo(CodeLineNum-1)

def CurrFiles(curr_dir = '.', ext = 'step_*'):
  '''Find the current directory file'''
  for i in glob.glob(os.path.join(curr_dir, ext)):
    yield i

def RemoveFiles(rootdir, ext, show = False):
    '''Deletes the current directory file'''
    for i in CurrFiles(rootdir, ext):
        if show:
            pass
        os.remove(i)

############################################################################################
def main(top_name, gro_name, em_mdp_name, md_mdp_name, target_mol_name, mut_resi_name,
     candi_resi_name, sub_resi_name, switch_num, reactive_site, new_atom_list, remove_resi_name, run_init):
    # deletes tmp file
    if run_init == 1:
        RemoveFiles('.', 'step_*', show=True)
        RemoveFiles('.', 'md_*', show=True)
        RemoveFiles('.', 'em_*', show=True)
        # for linux system
        os.system('rm -rf run_forward_* >& /dev/null')
    ## read the mutation mol as list. Get the number of runs
    natoms, boxv, totalmoles = ReadGMXGro(gro_name)
    mut_mol_init_list = []
    for info in totalmoles:
        if info[0][0] == mut_resi_name:
            mut_mol_init_list.append(info)
    sub_mol_init_num = len(mut_mol_init_list)
    if switch_num > 0:
        run_number = math.ceil(sub_mol_init_num / switch_num)
    else:
        print("!!!!!  Error  !!!!!")
        print("Switch_num must to be an integer greater than zero")
        sys.exit()

    # cycle operation gromacs
    #run_init = 1
    #run_number = 2
    #'''
    while run_init <= run_number:
        print("Runing time: %s/%s " % (run_init,run_number))
        # copy gro and top file
        if run_init == 1:
            copyfile(gro_name,"step_0.gro")
            copyfile(top_name, "step_0.top")
        new_gro_name = "step_" + str(run_init - 1) + ".gro"
        new_top_name = "step_" + str(run_init - 1) + ".top"

        ## read the all mol and Separate the molecules as difference list
        natoms, boxv, totalmoles = ReadGMXGro(new_gro_name)
        mut_mol_list = []
        sub_mol_list = []
        remove_mol_list = []
        other_mol_list = []
        for info in totalmoles:
            if info[0][0] == mut_resi_name:
                mut_mol_list.append(info)
            if info[0][0] == sub_resi_name:
                sub_mol_list.append(info)
            if info[0][0] == remove_resi_name: 
                remove_mol_list.append(info)
            if info[0][0] != sub_resi_name and info[0][0] != mut_resi_name and info[0][0] != remove_resi_name:
                other_mol_list.append(info)

        # Random selection of mutated molecules
        sub_mol_num = len(mut_mol_list)
        if sub_mol_num >= switch_num:
            random_list = random.sample(list(range(0, int(sub_mol_num), 1)), switch_num)
        else:
            random_list = list(range(0, int(sub_mol_num), 1))
        # Delete the mutated molecule
        after_mut_mol_list = [mut_mol_list[i] for i in range(0, len(mut_mol_list), 1) if i not in random_list]
        # Delete the water moleclue
        after_remove_mol_list = [remove_mol_list[i] for i in range(0, len(remove_mol_list), 1) if i not in random_list]
        #print(after_mut_mol_list)

        # The mol list of mutation 被突变的分子list
        remain_mut_list = []
        for info in random_list:
            mol_mut = mut_mol_list[info]
            remain_mut_list.append(mol_mut)
        #
        #print(remain_mut_list)
        # Calculate the centroid coordinates of new atom list
        com_coord_new_atom = GetComCoord(new_atom_list)
        #  Calculate the all centroid coordinates of substituted residue
        sub_coord_list = []
        for info in sub_mol_list:
            coord_sub = GetComCoord(info)
            #print(coord_sub)
            sub_coord_list.append(coord_sub)
        new_sub_mol_list = sub_mol_list
        #print(sub_coord_list)

        # Calculate the nearest distance and replace the molecule
        new_remain_mut_list = []
        new_remain_sub_list = []
        for info in remain_mut_list:
            tmp_atom_info = []
            for info1 in info:
                info1[0] = candi_resi_name
                tmp_atom_info.append(info1)

            # Calculate the centroid coordinates of new atom list
            reactive_site_list = []
            for info1 in reactive_site:
                reactive_site_list.append(info[info1 - 1])
            #print(reactive_site_list)
            com_coord_reactive_atom = GetComCoord(reactive_site_list)
            tmp_distace = []
            for info4 in sub_coord_list:
                dist = GetComDist(com_coord_reactive_atom, info4)
                tmp_distace.append(dist)
            #print(tmp_distace)
            #print(len(tmp_distace))
            index_dist = tmp_distace.index(min(tmp_distace))
            min_mol = new_sub_mol_list[index_dist]
            #print(min_mol)
            coord_min_mol = GetComCoord(min_mol)
            target_mol = ReadGMXGro(target_mol_name)[2]
            #print(len(target_mol))
            target_mol_com = GetComCoord(target_mol[0])
            #print(target_mol)
            tmp_remain_sub_list = []
            for info in target_mol[0]:
                tmp_info = []
                new_traget_mol_coord = TranslationCoordinates(info[2:], target_mol_com, coord_min_mol, 0, 0, 0)
                tmp_info.append(info[0])
                tmp_info.append(info[1])
                for info1 in new_traget_mol_coord:
                    tmp_info.append(info1)
                tmp_remain_sub_list.append(tmp_info)
            new_remain_sub_list.append(tmp_remain_sub_list)
            #print(new_remain_sub_list)
            del new_sub_mol_list[index_dist]
            del sub_coord_list[index_dist]
            #print(len(new_sub_mol_list),len(sub_coord_list))
            #print(com_coord_reactive_atom)
            for info2 in new_atom_list:
                tmp_info2 = []
                new_coord = TranslationCoordinates(info2[2:], com_coord_new_atom, com_coord_reactive_atom, 0.05, 0.05, 0.05)
                #print(new_coord)
                tmp_info2.append(info2[0])
                tmp_info2.append(info2[1])
                for info3 in new_coord:
                    tmp_info2.append(info3)
                #print(tmp_info2)
                tmp_atom_info.append(tmp_info2)
            new_remain_mut_list.append(tmp_atom_info)
        #print(new_remain_sub_list)
        #print(new_remain_mut_list)
        #print(len(after_mut_mol_list),len(new_sub_mol_list), len(new_remain_mut_list), len(new_remain_sub_list),len(other_mol_list))
        # Count the number of atoms of gro file
        total_atom_num = 0
        for info in after_mut_mol_list:
            for info1 in info:
                #print(info1)
                total_atom_num += 1
        #print(total_atom_num)
        for info in new_sub_mol_list:
            for info1 in info:
                #print(info1)
                total_atom_num += 1
        #print(total_atom_num)
        for info in new_remain_mut_list:
            #print(len(info))
            for info1 in info:
                #print(info1)
                total_atom_num += 1
        #print(total_atom_num)
        #print(len(new_remain_sub_list))
        for info in new_remain_sub_list:
            #print(len(info))
            for info1 in info:
                #print(info1)
                total_atom_num += 1
        #print(total_atom_num)
        for info in other_mol_list:
            for info1 in info:
                #print(info1)
                total_atom_num += 1
        for info in after_remove_mol_list: 
            for info1 in info:
                #print(info1)
                total_atom_num += 1
        #print(total_atom_num)
        #############################################################################
        # Create a new gro file
        gro_file = open("step_" + str(run_init) + ".gro", 'w+')
        now_time = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        gro_file.write("%s\n" % ('Generated by Jianchuan Liu  XHU\'s code on ' + str(now_time)))
        gro_file.write("%5d\n" % total_atom_num)
        # get the information of after_mut_mol_list
        all_resname = []
        resname, resid, natom = CreateNewGro(gro_file, after_mut_mol_list, 1, 1)
        all_resname += resname
        #  write the information of new_sub_mol_list
        resname, resid, natom = CreateNewGro(gro_file, new_sub_mol_list, resid, natom)
        all_resname += resname
        #  write the information of after_remove_mol_list
        resname, resid, natom = CreateNewGro(gro_file, after_remove_mol_list, resid, natom)
        all_resname += resname
        #  write the information of other_mol_list
        resname, resid, natom = CreateNewGro(gro_file, other_mol_list, resid, natom)
        all_resname += resname
        #  write the information of new_remain_mut_list
        resname, resid, natom = CreateNewGro(gro_file, new_remain_mut_list, resid, natom)
        all_resname += resname
        #  write the information of new_remain_sub_list
        resname, resid, natom = CreateNewGro(gro_file, new_remain_sub_list, resid, natom)
        all_resname += resname
        # write gro
        gro_file.write("%10.5f%10.5f%10.5f\n" % (boxv[0],boxv[1],boxv[2]))
        gro_file.close()
        #############################################################################
        #  Create a new top file
        top_file = open("step_" + str(run_init) + ".top", 'w+')
        line = GetLineNum(new_top_name, "molecules")[0][0]
        n = 1
        while n <= line:
            line_data = linecache.getline(new_top_name, n).strip('\n')
            top_file.write("%s\n" % line_data)
            n += 1
        for info in all_resname:
            top_file.write("%5s%5d\n" % (info,1))
        top_file.close()
        #############################################################################
        # run gromacs program
        RunGmx(run_init, em_mdp_name, md_mdp_name, 2023)
        CodeLineNum = sys._getframe().f_lineno
        GmxJudge(CodeLineNum, "md_" + str(run_init) + ".gro")
        #############################################################################
        # remove and create directory
        os.system('mkdir run_forward_%s >& /dev/null' % run_init)
        os.system('cp step_%s* ./run_forward_%s >& /dev/null' % (run_init, run_init))
        os.system('mv em_%s* ./run_forward_%s >& /dev/null' % (run_init, run_init))
        os.system('mv md_%s* ./run_forward_%s >& /dev/null' % (run_init, run_init))
        os.system("rm step_%d* >& /dev/null" % (run_init - 1))
        RemoveFiles('.', '#*#', show=True)
        run_init += 1
    os.system("cp run_forward_%s/md_%s.gro Final.gro >& /dev/null" % (run_init, run_init - 1))
    os.system("mv step_%s.top Final.top >& /dev/null" % (run_init - 1))
#'''
########################################################
if __name__ == '__main__':
    '''
    '''
    # groamcs gro file name
    gro_name = "init.gro"
    # gromacs  topology file name
    top_name = "top.top"
    # gromacs mdp file name
    em_mdp_name = "minim.mdp"
    md_mdp_name = "md.mdp"
    # single HCO3 gro file name
    target_mol_name = "HCO3.gro"
    # mutation residue name
    mut_resi_name = "DDA"
    # candidate residue name
    candi_resi_name = "DDP"
    # substituted residue name
    sub_resi_name = "P1P" # ["P1P", "P1N"]
    remove_resi_name = "P1N"
    ########################################
	### rerun No.
    run_init = 93
    # switching number of mol at unit simuation time (Must be an integer without a decimal point)
    switch_num = 100
    # the atom index list of Surfactants which atoms add hydrogen atoms
    reactive_site = [3]
    # Added the information of new atom and modify the residue name
    new_atom_list = [['DDP', 'H34', 0, 0, 0]]

    main(top_name, gro_name, em_mdp_name, md_mdp_name, target_mol_name, mut_resi_name,
         candi_resi_name, sub_resi_name, switch_num, reactive_site, new_atom_list, remove_resi_name, run_init)
## The End
