#!/usr/bin/env python3
# Edited by Akanksha Sachan, 2024
# Author: Lingfei Wang, 2022, 2023. All rights reserved.
import argparse
import itertools
import json
import logging
import sys
from collections import Counter
from functools import reduce
from operator import add
from os import listdir
from os.path import exists as pexists
from os.path import isdir
from os.path import join as pjoin

import numpy as np
import pandas as pd
from dictys.traj import point, trajectory

parser = argparse.ArgumentParser(
    description="Validates makefile and input data of network inference pipeline.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dir_data", type=str, default="data", help="Directory of input data folder."
)
parser.add_argument(
    "--dir_makefiles",
    type=str,
    default="makefiles",
    help="Directory of makefiles folder.",
)
parser.add_argument(
    "-c",
    action="store_const",
    default=False,
    const=True,
    help="Continue validation as much as possible except for critical errors.",
)

args = parser.parse_args()
dirdata = args.dir_data
dirmakefiles = args.dir_makefiles
nerr = 0 if args.c else None

if not isdir(dirmakefiles):
    raise FileNotFoundError(
        "Cannot find makefiles folder at {}. Please use option --dir_makefiles to specify makefiles folder.".format(
            dirmakefiles
        )
    )
if not isdir(dirdata):
    raise FileNotFoundError(
        "Cannot find input data folder at {}. Please use option --dir_data to specify input data folder.".format(
            dirdata
        )
    )

# Detect whether joint profiles from makefile
with open(pjoin(dirmakefiles, "config.mk"), "r") as f:
    s = f.readlines()
s = [x.strip() for x in s if x.startswith("JOINT=")][-1]
s = s[len("JOINT=") :]
if s not in {"0", "1"}:
    raise ValueError(
        "Invalid JOINT variable in {}. Only accepts: 0 for separate quantification of single-cell transcriptome and chromatin accessibility, 1 for joint quantification of single-cell transcriptome and chromatin accessibility.".format(
            pjoin(dirmakefiles, "config.mk")
        )
    )
isjoint = s == "1"
print(f"Joint profile: {isjoint}")

# Gene & cell names from RNA
namec_rna = pd.read_csv(
    pjoin(dirdata, "expression.tsv.gz"), header=0, index_col=0, sep="\t", nrows=1
)
namec_rna = np.array(namec_rna.columns)
t1 = [x[0] for x in dict(Counter(namec_rna)).items() if x[1] > 1][:3]
if len(t1) > 0:
    s = (
        "Duplicate cell names found in expression.tsv.gz. First three duplicate cell names: "
        + ", ".join(t1)
    )
    if nerr is None:
        raise ValueError(s)
    else:
        logging.error(s)
        nerr += 1
snamec_rna = set(namec_rna)
print(f"Found {len(namec_rna)} cells with RNA profile")
if len(namec_rna) < 100:
    s = "<100 cells found with RNA profile in expression.tsv.gz"
    if nerr is None:
        raise ValueError(s)
    else:
        logging.error(s)
        nerr += 1
nameg = pd.read_csv(
    pjoin(dirdata, "expression.tsv.gz"), header=0, index_col=0, sep="\t", usecols=[0, 1]
)
nameg = np.array(nameg.index)
t1 = [x[0] for x in dict(Counter(nameg)).items() if x[1] > 1][:3]
if len(t1) > 0:
    s = (
        "Duplicate gene names found in expression.tsv.gz. First three duplicate gene names: "
        + ", ".join(t1)
    )
    if nerr is None:
        raise ValueError(s)
    else:
        logging.error(s)
        nerr += 1
snameg = set(nameg)
print(f"Found {len(nameg)} genes with RNA profile")
if len(nameg) < 100:
    s = "<100 genes found with RNA profile in expression.tsv.gz"
    if nerr is None:
        raise ValueError(s)
    else:
        logging.error(s)
        nerr += 1

# Cell names from ATAC
try:
    namec_atac = listdir(pjoin(dirdata, "bams"))
    namec_atac = np.array(
        [".".join(x.split(".")[:-1]) for x in namec_atac if x.endswith(".bam")]
    )
except FileNotFoundError as e:
    if nerr is None:
        raise e
    else:
        logging.error(str(e))
        nerr += 1
        logging.warning(
            "Using RNA cell names for ATAC cell names for validations below."
        )
        namec_atac = namec_rna
snamec_atac = set(namec_atac)
print(f"Found {len(namec_atac)} cells with ATAC profile")
t1 = snamec_rna - snamec_atac
if isjoint and len(t1) > 0:
    s = (
        "Not all cells with RNA profile in expression.tsv.gz has a bam file in bams folder for the joint profiling dataset. First three cells missing: "
        + ", ".join(list(t1)[:3])
    )
    if nerr is None:
        raise FileNotFoundError(s)
    else:
        logging.error(s)
        nerr += 1

#############################################
# Dynamic GRN inference checks
#############################################

if not pexists(pjoin(dirmakefiles, "dynamic.mk")) or not pexists(
    pjoin(dirdata, "traj_node.h5")
):
    logging.warning(
        "Cannot find dynamic.mk or traj_node.h5. Skipping dynamic network inference checks."
    )
else:
    # For RNA
    traj = trajectory.from_file(pjoin(dirdata, "traj_node.h5"))
    pt_rna = point.from_file(traj, pjoin(dirdata, "traj_cell_rna.h5"))
    coord_rna = pd.read_csv(
        pjoin(dirdata, "coord_rna.tsv.gz"), header=0, index_col=0, sep="\t"
    )
    t1 = list(set(coord_rna.index) - snamec_rna)
    if len(t1) > 0:
        s = (
            "Low dimensional RNA cells in coord_rna.tsv.gz cannot be found in expression.tsv.gz. First three missing cells: "
            + ", ".join(t1)
        )
        if nerr is None:
            raise ValueError(s)
        else:
            logging.error(s)
            nerr += 1
    if len(coord_rna) != len(pt_rna):
        s = "Inconsistent RNA cell count between coord_rna.tsv.gz and traj_cell_rna.h5"
        if nerr is None:
            raise ValueError(s)
        else:
            logging.error(s)
            nerr += 1
    # For ATAC
    if not isjoint:
        pt_atac = point.from_file(traj, pjoin(dirdata, "traj_cell_atac.h5"))
        coord_atac = pd.read_csv(
            pjoin(dirdata, "coord_atac.tsv.gz"), header=0, index_col=0, sep="\t"
        )
        t1 = list(set(coord_atac.index) - snamec_atac)[:3]
        if len(t1) > 0:
            s = (
                "Low dimensional ATAC cells in coord_atac.tsv.gz cannot be found in bams folder. First three missing cells: "
                + ", ".join(t1)
            )
            if nerr is None:
                raise ValueError(s)
            else:
                logging.error(s)
                nerr += 1
        if len(coord_atac) != len(pt_atac):
            s = "Inconsistent ATAC cell count between coord_atac.tsv.gz and traj_cell_atac.h5"
            if nerr is None:
                raise ValueError(s)
            else:
                logging.error(s)
                nerr += 1

if nerr is not None and nerr > 0:
    raise RuntimeError(f"Found {nerr} error(s) in total.")
