import os
import sys
import time
import argparse

import numpy as np
from multiprocessing import Process, Array, Value

import nengo

from _spaun.configurator import cfg
from _spaun.vocabulator import vocab
from _spaun.experimenter import experiment
from _spaun.loggerator import logger
from _spaun.utils import (
    get_probe_data_filename,
    validate_num_gpus,
    build_and_run_spaun_network,
)
from _spaun.presets import stim_presets, cfg_presets

# ----- Defaults -----
def_dim = 512
def_seq = "A"
def_i = ""

# def_seq = "A1[1]?XXA1[22]?XX"
# def_seq = "{A1[R]?X:5}"
# def_seq = "{A3[{R:7}]?{X:8}:5}"
# def_seq = "{A3[{R:7}]?{X:8}:160}"
# def_seq = "A3[{R:7}]?{X:8}"
# def_seq = "%I1+I2%MP1.5[123]?XXXMP1.8[123]?XXX"
# def_i = "I1: 0.5*POS1 + 0.5*VIS*FOR, TASK*M + DEC*FWD;" + \
#         "I2: 0.5*POS1 + 0.5*VIS*EIG, TASK*M + DEC*REV"

# ----- Add current directory to system path ---
cfg.cwd = os.getcwd()

# ----- Parse arguments -----
parser = argparse.ArgumentParser(description="Script for running Spaun.")

parser.add_argument(
    "-d", type=int, default=def_dim,
    help="Number of dimensions to use for the semantic pointers.")
parser.add_argument(
    "--modules", type=str, default=None,
    help="A string of characters that determine what Spaun modules to " +
         "include when building Spaun: \n" +
         "S: Stimulus and monitor modules\n" +
         "V: Vision module\n" +
         "P: Production system module\n" +
         "R: Reward system module\n" +
         "E: Encoding system module\n" +
         "W: Working memory module\n" +
         "T: Transformation system module\n" +
         "D: Decoding system module\n" +
         "M: Motor system module\n" +
         "I: Instruction processing module\n" +
         "E.g. For all modules, provide \"SVPREWTDMI\". Note: Provide a \"-\" " +
         "as the first character to exclude all modules listed. E.g. To " +
         "exclude instruction processing module, provide \"-I\". ")

parser.add_argument(
    "-t", type=float, default=-1,
    help=("Simulation run time in seconds. If undefined, will be estimated" +
          " from the stimulus sequence."))
parser.add_argument(
    "-n", type=int, default=1,
    help="Number of batches to run (each batch is a new model).")

parser.add_argument(
    "-s", type=str, default=def_seq,
    help="Stimulus sequence. Use digits to use canonical digits, prepend a " +
         "'#' to a digit to use handwritten digits, a '[' for the open " +
         "bracket, a ']' for the close bracket, and a 'X' for each expected " +
         "motor response. e.g. A3[1234]?XXXX or A0[#1]?X")
# Stimulus formats:
# Special characters - A [ ] ?
# To denote Spaun stereotypical numbers: 0 1 2 3 4 5 6 7 8 9
# To denote spaces for possible answers: X
# To denote specific image classes: #0 or #100, (either a # or non-digit will
#                                                partition numbers)
# To denote a image chosen using an array index: <1000>
# To denote random numbers chosen without replacement: N
# To denote random numbers chosen with replacement: R
# To denote "reverse" option for memory task: B
# To denote matched random digits (with replacement): a - z (lowercase char)
# To denote forced blanks: .
# To denote changes in given instructions (see below): %INSTR_STR%
# Note:
#     Stimulus string can be duplicated using the curly braces in the format:
#     {<STIM_STR>:<DUPLICATION AMOUNT>}, e.g.,
#     {A3[RRR]?XXX:10}
parser.add_argument(
    "-i", type=str, default=def_i,
    help="Instructions event sequence. Use the following format to provide " +
         "customized instructions to spaun (which can then be put into the " +
         "stimulus string using \"%%INSTR_KEYN+INSTR_KEYM%%\": " +
         "\"INSTR_KEY: ANTECEDENT_SP_STR, CONSEQUENCE_SP_STR; ...\"" +
         "e.g. \"I1: TASK*INSTR + VIS*ONE, TRFM*POS1*THR\", and the stimulus " +
         "string: \"%%I1+I2%%A0[0]?XX\"")
# Note: For sequential position instructions, instruction must be encoded with
#       POS sp. E.g. I1: POS1+VIS*ONE, TASK*C
parser.add_argument(
    "--stim_preset", type=str, default="",
    help="Stimulus (stimulus sequence and instruction sequence pairing) to " +
         "use for Spaun stimulus. Overrides -s and -i command line options " +
         "if they are provided.")

parser.add_argument(
    "-b", type=str, default="ref",
    help="Backend to use for Spaun. One of [\"ref\", \"ocl\", \"spinn\"]")
parser.add_argument(
    "--data-dir", type=str, default=os.path.join(cfg.cwd, "data"),
    help="Directory to store output data.")
parser.add_argument(
    "--no-probes", action="store_true",
    help="Supply to disable probes.")
parser.add_argument(
    "--probeio", action="store_true",
    help="Supply to generate probe data for spaun inputs and outputs." +
         "(recorded in a separate probe data file)")
parser.add_argument(
    "--seed", type=int, default=-1,
    help="Random seed to use.")
parser.add_argument(
    "--showgrph", action="store_true",
    help="Supply to show graphing of probe data.")
parser.add_argument(
    "--savegrph", action="store_true",
    help="Supply to save graphed probed data.")
parser.add_argument(
    "--showanim", action="store_true",
    help="Supply to show animation of probe data.")
parser.add_argument(
    "--showiofig", action="store_true",
    help="Supply to show Spaun input/output figure.")
parser.add_argument(
    "--tag", type=str, default="",
    help="Tag string to apply to probe data file name.")
parser.add_argument(
    "--enable-cache", action="store_true",
    help="Supply to use nengo caching system when building the nengo model.")

parser.add_argument(
    "--multi-process", action="store_true",
    help="Supply to split model across multiple processes.")

parser.add_argument(
    "--ocl", action="store_true",
    help="Supply to use the OpenCL backend (will override -b).")
parser.add_argument(
    "--ocl-platform", type=int, default=0,
    help=("OCL Only: List index of the OpenCL platform to use. OpenCL " +
          " backend can be listed using \"pyopencl.get_platforms()\""))
parser.add_argument(
    "--ocl-device", type=int, default=0,
    help=("OCL Only: List index of the device on the OpenCL platform to use." +
          " OpenCL devices can be listed using " +
          "\"pyopencl.get_platforms()[X].get_devices()\" where X is the index " +
          "of the plaform to use."))
parser.add_argument(
    "--ocl-profile", action="store_true",
    help="Supply to use NengoOCL profiler.")

parser.add_argument(
    "--spinn", action="store_true",
    help="Supply to use the SpiNNaker backend (will override -b).")

parser.add_argument(
    "--nengo-gui", action="store_true",
    help="Supply to use the nengo_viz vizualizer to run Spaun.")

parser.add_argument(
    "--config", type=str, nargs="*",
    help="Use to set the various parameters in Spaun's configuration. Takes" +
         " arguments in list format. Each argument should be in the format" +
         " ARG_NAME=ARG_VALUE. " +
         "\nE.g. --config sim_dt=0.002 mb_gate_scale=0.8 " +
         "\"raw_seq_str='A1[123]?XX'\"" +
         "\nNOTE: Will override all other options that set configuration" +
         " options (i.e. --seed, --d, --s)" +
         "\nNOTE: Use quotes (\") to encapsulate strings if you encounter" +
         " problems.")
parser.add_argument(
    "--config-presets", type=str, nargs="*",
    help="Use to provide preset configuration options (which can be " +
         "individually provided using --config). Appends to list of " +
         "configuration options provided through --config.")

parser.add_argument(
    "--debug", action="store_true",
    help="Supply to output debug stuff.")

args = parser.parse_args()

# ----- Nengo RC Cache settings -----
# Disable cache unless seed is set (i.e. seed > 0) or if the "--enable_cache"
# option is given
if args.seed > 0 or args.enable_cache:
    print("USING CACHE")
    nengo.rc.set("decoder_cache", "enabled", "True")
else:
    print("NOT USING CACHE")
    nengo.rc.set("decoder_cache", "enabled", "False")

# ----- Backend Configurations -----
cfg.backend = args.b
if args.ocl:
    cfg.backend = "ocl"
if args.spinn:
    cfg.backend = "spinn"

print("BACKEND: %s" % cfg.backend.upper())

# ----- Multi-process Configurations ------
cfg.multi_process = args.multi_process

# ----- Stimulus sequence settings -----
if args.stim_preset in stim_presets:
    stim_seq_str, instr_seq_str = stim_presets[args.stim_preset]
else:
    stim_seq_str = args.s
    instr_seq_str = args.i

# ----- Gather configuration (from --config and --config_presets) settings ----
config_list = []
if args.config is not None:
    config_list += args.config

if args.config_presets is not None:
    for preset_name in args.config_presets:
        if preset_name in cfg_presets:
            config_list += cfg_presets[preset_name]

# ----- Batch runs -----
for n in range(args.n):
    print("\n======================== RUN %i OF %i ========================" %
          (n + 1, args.n))

    # ----- Seeeeeeeed -----
    timestamp = time.time()

    if args.seed < 0:
        seed = int(timestamp* 1000000) % (2 ** 32)
    else:
        seed = args.seed

    cfg.set_seed(seed)
    print("MODEL SEED: %i" % cfg.seed)

    # ----- Model Configurations -----
    vocab.sp_dim = args.d
    cfg.data_dir = args.data_dir

    # Parse --config options
    if len(config_list) > 0:
        print("USING CONFIGURATION OPTIONS: ")
        for cfg_options in config_list:
            cfg_opts = cfg_options.split("=")
            cfg_param = cfg_opts[0]
            cfg_value = cfg_opts[1]
            if hasattr(cfg, cfg_param):
                print("  * cfg: " + str(cfg_options))
                setattr(cfg, cfg_param, eval(cfg_value))
            elif hasattr(experiment, cfg_param):
                print("  * experiment: " + str(cfg_options))
                setattr(experiment, cfg_param, eval(cfg_value))
            elif hasattr(vocab, cfg_param):
                print("  * vocab: " + str(cfg_options))
                setattr(vocab, cfg_param, eval(cfg_value))

    # ----- Check if data folder exists -----
    if not(os.path.isdir(cfg.data_dir) and os.path.exists(cfg.data_dir)):
        raise RuntimeError("Data directory \"%s\"" % (cfg.data_dir) +
                           " does not exist. Please ensure the correct path" +
                           " has been specified.")

    # ----- Spaun imports -----
    from _spaun.utils import get_total_n_neurons
    from _spaun.spaun_main import Spaun

    from _spaun.modules.stim import stim_data
    from _spaun.modules.vision import vis_data
    from _spaun.modules.motor import mtr_data

    # ----- Enable debug logging -----
    if args.debug:
        nengo.log("debug")

    # ----- Experiment and vocabulary initialization -----
    experiment.initialize(stim_seq_str, stim_data.get_image_ind,
                          stim_data.get_image_label,
                          cfg.mtr_est_digit_response_time, instr_seq_str,
                          cfg.rng)
    vocab.initialize(stim_data.stim_SP_labels, experiment.num_learn_actions,
                     cfg.rng)
    vocab.initialize_mtr_vocab(mtr_data.dimensions, mtr_data.sps)
    vocab.initialize_vis_vocab(vis_data.dimensions, vis_data.sps)

    # ----- Spaun module configuration -----
    if args.modules is not None:
        used_modules = cfg.spaun_modules
        arg_modules = args.modules.upper()

        if arg_modules[0] == "-":
            used_modules = \
                "".join([s if s not in arg_modules else ""
                         for s in used_modules])
        else:
            used_modules = arg_modules

        cfg.spaun_modules = used_modules

    # ----- Configure output log files -----
    cfg.probe_data_filename = get_probe_data_filename(suffix=args.tag)

    # ----- Initalize looger and write header data -----
    logger.initialize(cfg.data_dir, cfg.probe_data_filename[:-4] + "_log.txt")

    logger.write("# Spaun Command Line String:\n")
    logger.write("# -------------------------\n")
    logger.write("# python " + " ".join(sys.argv) + "\n")
    logger.write("#\n")

    cfg.write_header()
    experiment.write_header()
    vocab.write_header()
    logger.flush()

    # ----- Raw stimulus seq -----
    print("RAW STIM SEQ: %s" % (str(experiment.raw_seq_str)))

    # ----- Spaun proper -----
    spaun_networks = Spaun()

    # ----- Validate multi-process resource requirements -----
    if cfg.multi_process and cfg.use_opencl:
        validate_num_gpus(len(spaun_networks) + args.ocl_device, args.ocl_platform)

    # ----- Display stimulus seq -----
    print("PROCESSED RAW STIM SEQ: %s" % (str(experiment.raw_seq_list)))
    print("STIMULUS SEQ: %s" % (str(experiment.stim_seq_list)))

    # ----- Neuron count debug -----
    neuron_counts = {}
    conn_counts = 0

    for net in spaun_networks:
        for module_name in ["vis", "ps", "reward", "bg", "thal", "enc",
                            "mem", "trfm", "instr", "dec", "mtr"]:
            if hasattr(net, module_name):
                if module_name not in neuron_counts:
                    neuron_counts[module_name] = 0
                neuron_counts[module_name] += \
                    get_total_n_neurons(getattr(net, module_name))
            # elif hasattr(net, "id_str") and net.id_str == module_name:
            #     if module_name not in neuron_counts:
            #         neuron_counts[module_name] = 0
            #     neuron_counts[module_name] += \
            #         get_total_n_neurons(net)
        conn_counts += len(net.all_connections)

    total_neuron_count = 0
    neuron_count_str = ""

    for module_name, count in neuron_counts.items():
        total_neuron_count += count
        neuron_count_str += f"\n- {module_name:6} n_neurons: {count}"

    neuron_count_str = f"MODEL N_NEURONS:  {total_neuron_count}" + neuron_count_str
    print(neuron_count_str)

    # ----- Connections count debug -----
    print(f"MODEL N_CONNECTIONS: {conn_counts}")

    # ----- Calculate runtime -----
    # Note: Moved up here so that we have data to disable probes if necessary
    runtime = args.t if args.t > 0 else experiment.get_est_simtime()

    # ----- Build and run the Spaun model -----
    buildtimes = Array("f", [0] * (len(spaun_networks)))
    walltimes = Array("f", [0] * (len(spaun_networks)))
    if len(spaun_networks) > 1:
        pids = []
        for proc_num, net in enumerate(spaun_networks):
            p = Process(
                    target=build_and_run_spaun_network,
                    args=(net, args, runtime, buildtimes, walltimes, proc_num)
                )
            p.start()
            pids.append(p)

        for p in pids:
            p.join()
    else:
        build_and_run_spaun_network(net, args, runtime, buildtimes, walltimes)

    # Special case where all build times are negative. This indicates that NengoGUI
    # was instantiated, so exit the script here.
    if np.allclose(buildtimes, -1):
        sys.exit()

    # Reset the experiment
    experiment.reset()

    if args.no_probes and not (args.showanim or args.showiofig or args.probeio):
        logger.write("\n# run_spaun.py was not instructed to record probe " +
                     "data.")

    # Close output logging file
    logger.close()

    # ----- Write runtime data -----
    runtime_filename = os.path.join(cfg.data_dir, "runtimes.txt")
    rt_file = open(runtime_filename, "a")
    rt_file.write("# ---------- TIMESTAMP: %i -----------\n" % timestamp)
    rt_file.write("Backend: %s | Num neurons: %i | Tag: %s | Seed: %i\n" %
                  (cfg.backend, total_neuron_count, args.tag, cfg.seed))
    if args.config is not None:
        rt_file.write("Config options: %s\n" % (str(args.config)))
    if len(spaun_networks) > 1:
        rt_file.write("Spaun networks: ")
        rt_file.write(", ".join([f"{net.label}" for net in spaun_networks]))
        rt_file.write("\n")

        rt_file.write("Build times: ")
        rt_file.write(", ".join([f"{t:.3f}s" for t in buildtimes]))
        rt_file.write("\n")

        rt_file.write(f"Model sim time: {runtime:.3f}s\n")

        rt_file.write("Sim wall times: ")
        rt_file.write(", ".join([f"{t:.3f}s" for t in walltimes]))
        rt_file.write("\n")
    else:
        rt_file.write(f"Build time: {buildtimes[0]:.3f}s | ")
        rt_file.write(f"Model sim time: {runtime:.3f}s | ")
        rt_file.write(f"Sim wall time: {walltimes[0]:.3f}s\n")
    rt_file.close()

    # ----- Cleanup -----
    model = None
    sim = None
    probe_data = None
