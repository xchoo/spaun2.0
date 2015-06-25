"""
A script for submitting a spaun simulation using nengo_mpi (or plain
nengo) to the GPC job queue.

Creates a new directory for each experiment in ``experiments_dir'',
which is labelled by date and time the experiment was started.
Launches the script from that directory, and then writes the output
from running the script to ``results.txt'' in that directory.
Also creates a sym link called ``latest'' in ``experiments_dir'',
which points to the directoy of the most recently run experiment.

Can be used in three ways:

    1. loading a spaun network from a file and simulating it.
    2. building a spaun network and saving it to a file.
    3. building a spaun network and running it immediately, without saving.

Example:

Submits a spaun job that we expect to take 30 minutes, using 4 hardware nodes
(so 4 * 8 = 32 processes), and saving the resulting network to file to the
default save location:

python qsubmit.py -w 0:30:00 -n 4 --save None

"""

import string
import os
import subprocess
import argparse
import datetime

default_file_location = (
    "/scratch/c/celiasmi/e2crawfo/spaun_experiments/spaun.net")

parser = argparse.ArgumentParser(
    description="Run nengo mpi simulation using GPC queue.")

parser.add_argument(
    '--save', nargs='?', const=default_file_location,
    help="Supply this arg to have the code build the spaun network and "
         "save it to a file, but not simulate the network. Cannot supply both "
         "--save and --load. If no filename is supplied, a default of %s "
         "is used." % default_file_location)

parser.add_argument(
    '--load', nargs='?', const=default_file_location,
    help="Supply this arg to load the spaun model from the file with "
         "given name instead of building the spaun model. The number of "
         "processors to use will be extracted from the file, and arguments "
         "specifying the number of processors are ignored. Cannot supply both "
         "--save and --load. If no filename is supplied, a default of %s "
         "is used." % default_file_location)

parser.add_argument(
    '-n', type=int, default=1, help="Number of hardware nodes.")

parser.add_argument(
    '-w', default="0:15:00",
    help="Upper bound on required wall time.")

parser.add_argument(
    '-o', default='',
    help="Arguments for the called program. Not used if "
         "loading network from a file.")

parser.add_argument(
    '--sim-length', type=float, default=1.0, dest="sim_length",
    help="Length of the simulation. Used only if loading "
         "network from a file.")

args = parser.parse_args()

load = args.load
save = args.save

if load and save:
    raise ValueError("--save and --load are mutually exclusive options.")

# processors per hardware node
ppn = 8

n_nodes = args.n

# set total number of processors
if load:
    with open(load, 'r') as f:
        line = f.readline()
        n_processors = int(line.split('|')[0])
else:
    n_processors = ppn * n_nodes

wall_time = args.w
script_args = args.o
sim_length = args.sim_length

experiments_dir = "/scratch/c/celiasmi/e2crawfo/spaun_experiments"
directory = experiments_dir + "/exp_"
date_time_string = str(datetime.datetime.now()).split('.')[0]
date_time_string = reduce(
    lambda y, z: string.replace(y, z, "_"),
    [date_time_string, ":", " ", "-"])
directory += date_time_string

if save:
    directory += "_save"
if load:
    directory += "_load"

directory += '_p_%d' % n_processors

if not os.path.isdir(directory):
    os.makedirs(directory)

submit_script_name = "submit_script.sh"
results = directory + "/results.txt"

spaun_directory = (
    '/home/c/celiasmi/e2crawfo/spaun2.0/')

spaun_script_location = (
    '/home/c/celiasmi/e2crawfo/spaun2.0/scripts/test_spaun_mpi.py')

mpi_worker_location = (
    '/home/c/celiasmi/e2crawfo/nengo_mpi/nengo_mpi/mpi_sim_worker')


def make_sym_link(target, name):
    try:
        os.remove(name)
    except OSError:
        pass

    os.symlink(target, name)

make_sym_link(directory, experiments_dir+'/latest')

with open(directory + '/' + submit_script_name, 'w') as outf:
    outf.write("#!/bin/bash\n")
    outf.write("# MOAB/Torque submission script for SciNet GPC\n")
    outf.write("#\n")

    if save:
        outf.write(
            "#PBS -l nodes=%d:ppn=%d,walltime=%s\n" % (1, ppn, wall_time))
    else:
        outf.write(
            "#PBS -l nodes=%d:ppn=%d,walltime=%s"
            "\n" % (n_nodes, ppn, wall_time))

    outf.write("#PBS -N spaun\n")
    outf.write("#PBS -m abe\n\n")

    outf.write("# load modules (must match modules used for compilation)\n")
    outf.write("module load intel/14.0.1\n")
    outf.write("module load python/2.7.8\n")
    outf.write("module load openmpi/intel/1.6.4\n")
    outf.write("module load cxxlibraries/boost/1.55.0-intel\n")
    outf.write("module load gcc/4.8.1\n")
    outf.write("module load use.own\n")
    outf.write("module load nengo\n\n")

    outf.write("cd %s\n" % directory)
    outf.write("cp ${PBS_NODEFILE} .\n\n")
    outf.write("cd %s\n" % spaun_directory)

    execution_line = ""

    if save:
        script_args += " --save %s -p %d " % (save, n_processors)
        execution_line = (
            "mpirun -np 1 python %s %s "
            "> %s" % (spaun_script_location, script_args, results))

        print ("Job will build the Spaun model using %s\n"
               "and save the network to %s without running a\n"
               "simulation." % (spaun_script_location, save))

    elif load:
        scripts_args = "%s %s" % (load, sim_length)
        execution_line = "mpirun -np %d --mca pml ob1 %s %s %f 0 > %s"
        execution_line = execution_line % (
            n_processors, mpi_worker_location, load, sim_length, results)

        print ("Job will load the Spaun model saved at %s\n"
               "and run it for %f seconds using %d nodes and\n"
               "%d processors per node (total of %d processors)."
               "" % (load, sim_length, n_nodes, ppn, n_processors))

    else:
        script_args += " -p %d --dir %s " % (n_processors, directory)
        execution_line = (
            "mpirun -np 1 --mca pml ob1 python %s --noprog %s "
            "> %s" % (spaun_script_location, script_args, results))

        print ("Job will build the Spaun model using %s\n"
               "and run a simulation using %d nodes and\n"
               "%d processors per node (total of %d processors)."
               "" % (spaun_script_location, n_nodes, ppn, n_processors))

    outf.write(execution_line)

os.chdir(directory)
job_id = subprocess.check_output(['qsub', submit_script_name])
job_id = job_id.split('.')[0]

open(directory + '/' + job_id, 'w').close()
print "\nJob ID: ", job_id

print "\nExecution line:"
print execution_line
