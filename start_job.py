'''
Created on Mar 25, 2014

@author: koher
'''
# -*- coding: utf-8 -*-
##!/usr/bin/env python
from numpy import arange,concatenate
from subprocess import Popen
from os import makedirs,getcwd
from os.path import exists
import datetime
from shutil import copy
from glob import glob
now = datetime.datetime.now()

#=======================================================================
# Parameter:
# (0) zm:        wenn die ZM bereits vorhanden ist und nur noch ausgewertet werden muss
#
# (1) Wenn zufaellige Adjazenzmatrizen generiert werden muessen, dann 
#     sind die folgenden Parameter einzugeben
#     graph = []
#     memory:    infektioese Periode
#     size:      Groesse der Adjazenzmatrix
#     prob:      Linkwahrscheinlichkeit beim Erdos-Renyi-Graphen
#     runtime:   Simulationsdauer
#     path:      Zielordner (default ist '/users/stud/koher/workspace/master/')
#     mode:      Infektionsmodell: SI, SIS, SIR
#
# (2) Wenn aus gegebenen Daten die Adjazenzmatrizen generiert werden sollen:
#     sind die folgenden Paramter obligatorisch
#     graph:     'sociopatterns_hypertext.dat' oder 'sexual_contacts.dat'
#     path
#     mode
#     memory
#=======================================================================
graph    = "sexual_contacts.dat"
size     = 0
prob     = 0
runtime  = 0
path     = "/users/stud/koher/master/python/" + now.strftime("%Y_%m_%d") + "/"
mode     = "aggregate"
memories = [0]
#memories = arange(100,2300,100)

#===============================================================================
# Vor jeder Simulation werden saemtliche .py Dateien vom workspace in einen 
# Ordner kopiert, der im Namen das Datum traegt: JJJJ_MM_DD
# Wenn die Dateien bzw. der Ordner schon existiert wird nichts gemacht.
#===============================================================================
if not exists(path):
    makedirs(path)
files = glob(getcwd()+"/*.py")
files.append(getcwd()+"/job")
for afile in files:
    copy(afile, path)

#===============================================================================
# Fuer alle Parameterkombinationen wird ein shell-skript gestartet ans Cluster
# abschickt. Das skript heisst job und uebernimmt alle gegebenen Parameter
#===============================================================================
for memory in memories:
    if (graph == "sociopatterns_hypertext.dat" or graph == "sexual_contacts.dat"):
        name = "m="+str(memory)+"_"+mode
        Popen('qsub -o %s -v MEMORY="%5i",PROB="%1.1e",SIZE="%5i",TIME="%5i",DIR="%s",MODE="%s",GRAPH="%s" -N %s job' %(path,memory,prob,size,runtime,path,mode,graph,name), shell=True)
        
    else:
        name = "m="+str(memory)+"_"+"p="+str(prob)+"_"+mode
        Popen('qsub -o %s -v MEMORY="%5i",PROB="%1.1e",SIZE="%5i",TIME="%5i",DIR="%s",MODE="%s",GRAPH="%s" -N %s job' %(path,memory,0,0,0,path,mode,"0",name), shell=True)
