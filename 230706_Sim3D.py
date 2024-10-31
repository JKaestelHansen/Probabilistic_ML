from math import gamma
from multiprocessing.sharedctypes import Value
from polychrom.hdf5_format import HDF5Reporter
from polychrom import simulation, starting_conformations, forces, forcekits, polymerutils, hdf5_format
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
from SMC_class import *
import h5py
import time
import openmm
import csv
import simtk

# files to be saved
Left_CTCF_file_path = 'leftCTCF.csv'
Right_CTCF_file_path = 'rightCTCF.csv'


"""Simulation parameters"""
# Locus and bead parameters
bp_per_bead = 750  # bp/bead
locus_size = 3 * 505000  # bp, TAD was 505kb and we add half that flanking each side
nrepeats = 50
dens = 0.2  # ~20% of the volume is occupied by the beads

# Guess for reasonable simulation time per block...this might be close to a second in real time but one would need to compare to data to figure that out for sure
nseconds_per_block = 20
blocksize = 2213 * nseconds_per_block
# update_bonds_every = 1 
# n_blocks_to_simulate = 30000 * 10  # estimated blocks to steady state * 10
"""""" """""" """""" """""" """"""

# Functions
# save simulation's CTCF 
# ELEPHANT: ADD TIME + # + merge 
# dataframe data file type, pandas
# use pandas instead of numpy
def saveCTCFPositions(sim, leftpositions, rightpositions, save_to_folder='.', extra=[]):
    Left_CTCF_file_path = f"{save_to_folder}/leftCTCF.csv"
    Right_CTCF_file_path = f"{save_to_folder}/rightCTCF.csv"
    sim.state = sim.context.getState(getPositions=True)
    coords = sim.state.getPositions(asNumpy=True)
    data = coords / simtk.unit.nanometer
    data = np.array(data, dtype=np.float32)


    with open(Left_CTCF_file_path, "a") as leftCTCFFile:
        # leftWriter = csv.writer(leftCTCFFile)
        for i in leftpositions:
            if i < data.shape[0]:
                savedata = pd.DataFrame([i]+ list(data[i]))
                if extra:
                    savedata = pd.concat([savedata, pd.DataFrame(extra)])
                savedata = savedata.T
                # savedata = np.concatenate([i], data[i]) # insert the 1d index to the first column
                # if extra:
                #     savedata = np.concatenate((savedata, extra))
                savedata.to_csv(leftCTCFFile, header=None)
    with open(Right_CTCF_file_path, "a") as rightCTCFFile:
        # rightWriter = csv.writer(rightCTCFFile)
        for i in rightpositions:
            if i < data.shape[0]:
                savedata = pd.DataFrame([i]+ list(data[i]))
                if extra:
                    savedata = pd.concat([savedata, pd.DataFrame(extra)])
                savedata = savedata.T
                savedata.to_csv(rightCTCFFile, header=None)
                


# Load the last configuration in the long simulation
block_nums_files = [int(i.split(":")[-1]) for i in hdf5_format.list_URIs(
    "/home/adu/levsha/PRIMES/23_5_5_initial_system_setup/test_427")]
data = polymerutils.fetch_block(
    "/home/adu/levsha/PRIMES/23_5_5_initial_system_setup/test_427", np.max(block_nums_files))
# Recalculate N so it matches (before it was 75 sites of 2*505kb now we use 50 sites of 3*750kb)
N = len(data)


"""Initial calculations"""
nbeads = round(locus_size / bp_per_bead)
N = nbeads * nrepeats
# calculate box length needed to thet the wanted density
box = (N / dens) ** 0.33

"""Initialize the simulation"""
"""1D Loop extrusion data load"""
path_to_folder_where_sims_are_to_be_stored = "/home/adu/levsha/PRIMES/23_5_10_loop_extrusion_sims/0807"

simname = "1D_trajectory"
result = polymerutils.hdf5_format.load_hdf5_file(
    f"{path_to_folder_where_sims_are_to_be_stored}/{simname}.h5"
)

myfile = h5py.File(
    f"{path_to_folder_where_sims_are_to_be_stored}/{simname}.h5", mode="r"
)

N = myfile.attrs["N"]
LEFNum = myfile.attrs["LEFNum"]
LEFpositions = myfile["positions"]
Nframes = LEFpositions.shape[0]
Leftpositions = myfile.attrs["Leftpositions"]
Rightpositions = myfile.attrs["Rightpositions"]
left_CTCF_position = myfile.attrs["left_CTCF_position"]
right_CTCF_position = myfile.attrs["right_CTCF_position"]

print(f"Leftpositions: {Leftpositions}")
print(f"left_CTCF_position: {left_CTCF_position}")

smcBondWiggleDist = 0.2
smcBondDist = 0.5
""""""


saveEveryBlocks = 50
#Nframes = saveEveryBlocks * n_blocks_to_simulate
restartSimulationEveryBlocks = 1000


# assertions for easy managing code below
assert (Nframes % restartSimulationEveryBlocks) == 0
assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
simInitsTotal = (Nframes) // restartSimulationEveryBlocks

# Set up the reporter that will save the simulation data
reporter = HDF5Reporter(
    folder=path_to_folder_where_sims_are_to_be_stored,
    max_data_length=10,  # save every 10 blocks
    # will overwrite anything that is already there (remember for if you want to redo a simulation)
    overwrite=True,
    check_exists=False,
)


milker = bondUpdater(LEFpositions)
step = 0  # starting block

# this step actually puts all bonds in and sets first bonds to be what they should be
for iteration in range(simInitsTotal):
    print(f"Total steps run: {step}")

    # simulation parameters are defined below
    a = simulation.Simulation(
        platform="CUDA",
        integrator="VariableLangevin",
        error_tol=0.01,
        GPU=(sys.argv[1] if len(sys.argv) > 1 else "0"),
        collision_rate=0.03,
        N=N,
        save_decimals=2,
        precision="mixed",
        PBCbox=[box, box, box],
        reporters=[reporter], # ELEPHANT can comment out to not store total polymer
        verbose=True,
    )

    ############################## New code ##############################
    a.set_data(data)  # loads a polymer, puts a center of mass at zero

    a.add_force(
        forcekits.polymer_chains(
            a,
            chains=[(0, None, 0)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.05,
            },
            angle_force_func=forces.angle_force,
            angle_force_kwargs={"k": 4},
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": 1.5,
                "radiusMult": 1,
            },
            except_bonds=True,
        )
    )
    kbond = a.kbondScalingFactor / (smcBondWiggleDist**2)
    bondDist = smcBondDist * a.length_scale
    activeParams = {"length": bondDist, "k": kbond}
    inactiveParams = {"length": bondDist, "k": 0}
    milker.setParams(activeParams, inactiveParams)
    # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(
        bondForce=a.force_dict["harmonic_bonds"], blocks=restartSimulationEveryBlocks)

    # If your simulation does not start, consider using energy minimization below
    # if iteration == 0:
    #     a.local_energy_minimization()
    # else:
    a._apply_forces()

    """
    store ctcf site position code: data = a.getdata
    save to file in the right folder
    """
    for i in range(restartSimulationEveryBlocks):
        step += 1
        timestep = i
        
        # ELEPHANT time as a parameter 
        if i % saveEveryBlocks == (saveEveryBlocks - 1):
            a.do_block(steps=blocksize)
        else:
            a.integrator.step(
                blocksize
            )  # do steps without getting the positions from the GPU (faster)
        if i < restartSimulationEveryBlocks - 1:
            # if i % update_bonds_every:
            curBonds, pastBonds = milker.step(
                a.context
            )  # this updates bonds. You can do something with bonds here
        
        saveCTCFPositions(a,
                          Leftpositions,
                          Rightpositions,
                          path_to_folder_where_sims_are_to_be_stored,
                          [int(iteration), int(timestep)])


    data = a.get_data()  # save data and step, and delete the simulation
    del a

    reporter.blocks_only = True  # Write output hdf5-files only for blocks

    # wait 200ms for sanity (to let garbage collector do its magic)
    time.sleep(0.2)

reporter.dump_data()
