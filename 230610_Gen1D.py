from SMC_class import *
import os
import h5py
from tqdm import tqdm
from polychrom import polymerutils

"""Simulation parameters"""
# simulation storage params
path_to_folder_where_sims_are_to_be_stored = "/home/adu/levsha/PRIMES/23_5_10_loop_extrusion_sims/0807"
simname = "1D_trajectory"



# Locus and bead parameters
bp_per_bead = 750  # bp/bead
locus_size = (3/2)*2 * 505000  # bp, TAD was 505kb and we add half that flanking each side
nrepeats = 50
dens = 0.2  # ~20% of the volume is occupied by the beads

# Guess for reasonable simulation time per block...this might be close to a second in real time but one would need to compare to data to figure that out for sure
nseconds_per_block = 20
blocksize = 2213 * nseconds_per_block
n_blocks_to_simulate = 30000 * 10 #estimated blocks to steady state * 10
"""""" """""" """""" """""" """"""

nbeads = int(locus_size / bp_per_bead)
left_CTCF_position = nbeads//3
right_CTCF_position = 2*nbeads//3
N = 100950 # number of beads in the configuration we are using

# ~150kb separation
sep_in_bp = 300_000
SEPARATION = sep_in_bp//bp_per_bead

#100kb before falling off
lifetime_in_bp = 150_000
LIFETIME = lifetime_in_bp//bp_per_bead

nlifetimes_to_simulate = 100
TRAJECTORY_LENGTH = nlifetimes_to_simulate*LIFETIME

stall_prob=1
release_prob=0

LEFNum = round(N // SEPARATION) # number of cohesin complexes


Leftpositions = [i*nbeads+left_CTCF_position for i in range(nrepeats)] 
Rightpositions = [i*nbeads+right_CTCF_position for i in range(nrepeats)] 


args = {}
args["ctcfRelease"] = {-1: dict(zip(Leftpositions,[release_prob for i in Leftpositions])), 1: dict(zip(Rightpositions,[release_prob for i in Rightpositions]))}
args["ctcfCapture"] = {-1: dict(zip(Leftpositions,[stall_prob for i in Leftpositions])), 1: dict(zip(Rightpositions,[stall_prob for i in Rightpositions]))}
args["N"] = N
args["LIFETIME"] = LIFETIME
args["LIFETIME_STALLED"] = LIFETIME  # no change in lifetime when stalled


occupied = np.zeros(N)
# set boundary conds on chains
occupied[0] = 1
occupied[-1] = 1
cohesins = []

for i in range(LEFNum):
    loadOne(cohesins, occupied, args)

with h5py.File(
    f"{path_to_folder_where_sims_are_to_be_stored}/{simname}.h5", mode="w"
) as myfile:
    dset = myfile.create_dataset(
        "positions", shape=(TRAJECTORY_LENGTH, LEFNum, 2), dtype=np.int32, compression="gzip"
    )
    steps = 500  # saving in 50 chunks because the whole trajectory may be large
    bins = np.linspace(0, TRAJECTORY_LENGTH, steps, dtype=int)  # chunks boundaries
    for st, end in tqdm(list(zip(bins[:-1], bins[1:]))):
        cur = []
        for i in range(st, end):
            translocate(cohesins, occupied, args)  # actual step of LEF dynamics
            positions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
            cur.append(positions)  # appending current positions to an array
        cur = np.array(cur)  # when we finished a block of positions, save it to HDF5
        dset[st:end] = cur
    myfile.attrs["N"] = N
    myfile.attrs["LEFNum"] = LEFNum
    myfile.attrs["Leftpositions"] = Leftpositions
    myfile.attrs["Rightpositions"] = Rightpositions
    myfile.attrs["left_CTCF_position"] = left_CTCF_position
    myfile.attrs["right_CTCF_position"] = right_CTCF_position