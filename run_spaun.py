import os
import sys
import time
import argparse

import nengo

from _spaun.configurator import cfg
from _spaun.vocabulator import vocab
from _spaun.experimenter import experiment
from _spaun.loggerator import logger
from _spaun.utils import get_probe_data_filename

# ----- Defaults -----
def_dim = 512
def_seq = 'A'
def_i = ''
def_mpi_p = 128

# ----- Spaun (character & instruction) presets -----
stim_presets = {}

# Standard Spaun stimulus presets
# TODO: Add in configuration options into presets as well?
stim_presets['copy_draw'] = ('A0[#1]?X', '')
stim_presets['copy_draw_mult'] = ('A0[#1#2#3]?XXX', '')
stim_presets['digit_recog'] = ('A1[#1]?XXX', '')
stim_presets['learning'] = ('A2?{X:30}', '')
stim_presets['memory_3'] = ('A3[123]?XXXX', '')
stim_presets['memory_4'] = ('A3[1234]?XXXX', '')
stim_presets['memory_7'] = ('A3[2567589]?XXXXXXXXX', '')
stim_presets['count_3'] = ('A4[5][3]?XXXXXX', '')
stim_presets['count_9'] = ('A4[0][9]?XXXXXXXXXXX', '')
stim_presets['count_3_list'] = ('A4[321][3]?XXXXXXX', '')
stim_presets['qa_kind'] = ('A5[123]K[3]?X', '')
stim_presets['qa_pos'] = ('A5[123]P[1]?X', '')
stim_presets['rvc_simple'] = ('A6[12][2][82][2][42]?XXXXX', '')
stim_presets['rvc_complex'] = ('A6[8812][12][8842][42][8862][62][8832]?XXXXX',
                               '')
stim_presets['induction_simple'] = ('A7[1][2][3][2][3][4][3][4]?X', '')
stim_presets['induction_incomplete'] = ('A7[1][2][3][2]?XX', '')
stim_presets['induction_ravens'] = ('A7[1][11][111][2][22][222][3][33]?XXXXX',
                                    '')

# Darpa adaptive motor presets
stim_presets['darpa_adapt_motor1'] = ('{A3[#4#2#7#5]?XXXX:8}', '')

# Darpa imagenet presets
stim_presets['darpa_imagenet1'] = ('{AC[#BOX_TURTLE][#BOX_TURTLE]?X' +
                                   'AC[#SEWING_MACHINE][#SEWING_MACHINE]?X' +
                                   'AC[#GUENON][#GUENON]?X' +
                                   'AC[#TIBETAN_TERRIER][#TIBETAN_TERRIER]?X' +
                                   'AC[#PERSIAN_CAT][#PERSIAN_CAT]?X:5}', '')
stim_presets['darpa_imagenet2'] = ('{AC[#BOX_TURTLE][#SEWING_MACHINE]?X' +
                                   'AC[#BOX_TURTLE][#GUENON]?X' +
                                   'AC[#BOX_TURTLE][#TIBETAN_TERRIER]?X' +
                                   'AC[#BOX_TURTLE][#PERSIAN_CAT]?X:5}', '')
stim_presets['darpa_imagenet3'] = ('{AC[#SEWING_MACHINE][#BOX_TURTLE]?X' +
                                   'AC[#SEWING_MACHINE][#GUENON]?X' +
                                   'AC[#SEWING_MACHINE][#TIBETAN_TERRIER]?X' +
                                   'AC[#SEWING_MACHINE][#PERSIAN_CAT]?X:5}',
                                   '')
stim_presets['darpa_imagenet4'] = ('{AC[#GUENON][#BOX_TURTLE]?X' +
                                   'AC[#GUENON][#SEWING_MACHINE]?X' +
                                   'AC[#GUENON][#TIBETAN_TERRIER]?X' +
                                   'AC[#GUENON][#PERSIAN_CAT]?X:5}', '')
stim_presets['darpa_imagenet5'] = ('{AC[#TIBETAN_TERRIER][#BOX_TURTLE]?X' +
                                   'AC[#TIBETAN_TERRIER][#SEWING_MACHINE]?X' +
                                   'AC[#TIBETAN_TERRIER][#GUENON]?X' +
                                   'AC[#TIBETAN_TERRIER][#PERSIAN_CAT]?X:5}',
                                   '')
stim_presets['darpa_imagenet6'] = ('{AC[#PERSIAN_CAT][#BOX_TURTLE]?X' +
                                   'AC[#PERSIAN_CAT][#SEWING_MACHINE]?X' +
                                   'AC[#PERSIAN_CAT][#GUENON]?X' +
                                   'AC[#PERSIAN_CAT][#TIBETAN_TERRIER]?X:5}',
                                   '')

# Darpa instruction following presets
stim_resp_i = 'I1: VIS*ONE, DATA*POS1*NIN;I2: VIS*TWO, DATA*POS1*EIG;' + \
              'I3: VIS*THR, DATA*POS1*SEV;I4: VIS*FOR, DATA*POS1*SIX;' + \
              'I5: VIS*FIV, DATA*POS1*FIV;I6: VIS*SIX, DATA*POS1*FOR;' + \
              'I7: VIS*SEV, DATA*POS1*THR;I8: VIS*EIG, DATA*POS1*TWO;' + \
              'I9: VIS*NIN, DATA*POS1*ONE;I0: VIS*ZER, DATA*POS1*ZER'
stim_presets['darpa_instr_stim_resp_2'] = \
    ('%I1+I2%A9{?1X?2X:5}%I3+I4%A9{?4X?3X:5}', stim_resp_i)
stim_presets['darpa_instr_stim_resp_3'] = \
    ('%I1+I2+I3%A9{?1X?2X?3X:5}%I4+I5+I6%A9{?6X?5X?4X:5}', stim_resp_i)
stim_presets['darpa_instr_stim_resp_4'] = \
    ('%I1+I2+I3+I4%A9{?1X?2X?3X?4X:5}%I0+I9+I8+I7%A9{?0X?9X?8X?7X:5}',
     stim_resp_i)
stim_presets['darpa_instr_stim_resp_5'] = \
    ('%I1+I2+I3+I4+I5%A9{?1X?2X?3X?4X?5X:5}' +
     '%I0+I9+I8+I7+I6%A9{?0X?9X?8X?7X?6X:5}', stim_resp_i)
stim_presets['darpa_instr_stim_resp_6'] = \
    ('%I1+I2+I3+I4+I5+I6%A9{?1X?2X?3X?4X?5X?6X:5}' +
     '%I0+I9+I8+I7+I6+I5%A9{?0X?9X?8X?7X?6X?5X:5}', stim_resp_i)

stim_task_i = 'I1: VIS*ONE, TASK*F;I2: VIS*TWO, TASK*C;' + \
              'I3: VIS*THR, TASK*M + DEC*REV; I4: VIS*FOR, TASK*W;' + \
              'I5: VIS*FIV, TASK*M; I6: VIS*SIX, TASK*V;' + \
              'I7: VIS*SEV, TASK*A;I8: VIS*EIG, TASK*REACT+STATE*DIRECT'
stim_presets['darpa_instr_stim_task_2'] = \
    ('%I1+I2%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX:5}' +
     '%I3+I4%{M3.[321]?XXXM4.[0]?X:5}', stim_task_i)
stim_presets['darpa_instr_stim_task_3'] = \
    ('%I1+I2+I3%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXXM3.[321]?XXX:5}' +
     '%I4+I5+I6%{M4.[0]?XM5.[123]?XXXM6.[13][3][12][2][11]?X:5}', stim_task_i)
stim_presets['darpa_instr_stim_task_4'] = \
    ('%I1+I2+I3+I4%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX' +
     'M3.[321]?XXXM4.[9]?X:5}%I3+I4+I5+I6%{M3.[987]?XXXM4[0]?X' +
     'M5.[123]?XXXM6.[13][3][12][2][11]?X:5}', stim_task_i)
stim_presets['darpa_instr_stim_task_5'] = \
    ('%I1+I2+I3+I4+I5%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX' +
     'M3.[321]?XXXM4.[9]?XM5.[123]?XXX:5}%I4+I5+I6+I7+I8%{M4[0]?X' +
     'M5.[123]?XXXM6.[13][3][12][2][11]?XM7.[123]P[3]?XM8.?1X?2X:5}',
     stim_task_i)
stim_presets['darpa_instr_stim_task_6'] = \
    ('%I1+I2+I3+I4+I5+I6%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX' +
     'M3.[321]?XXXM4.[9]?XM5.[123]?XXXM6.[39][9][38][8][37]?X:5}' +
     '%I3+I4+I5+I6+I7+I8%{M3.[876]?XXXM4[0]?XM5.[456]?XXX' +
     'M6.[13][3][12][2][11]?XM7.[123]P[3]?XM8.?1X?2X:5}', stim_task_i)

seq_task_i = 'I1: POS1, TASK*F;I2: POS2, TASK*C;' + \
             'I3: POS3, TASK*M + DEC*REV; I4: POS4, TASK*W;' + \
             'I5: POS5, TASK*M; I6: POS6, TASK*V;' + \
             'I7: POS7, TASK*A;I8: POS8, TASK*REACT+STATE*DIRECT'
stim_presets['darpa_instr_seq_task_2'] = \
    ('%I1+I2%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX:5}' +
     '%I3+I4%{MP3.[321]?XXXMP4.[0]?X:5}', seq_task_i)
stim_presets['darpa_instr_seq_task_3'] = \
    ('%I1+I2+I5%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX' +
     'MP5.[123]?XXX:5}%I3+I4+I6%{MP6.[21][1][24][4][26][6][28]?' +
     'XXMP4.[0]?XMP3.[321]?XXX:5}', seq_task_i)
stim_presets['darpa_instr_seq_task_4'] = \
    ('%I1+I2+I5+I7%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX' +
     'MP5.[123]?XXXMP7.[123]K[3]?X:5}%I3+I4+I6+I8%{MP8.?1X?2X' +
     'MP6.[21][1][24][4][26][6][28]?XXMP4.[0]?XMP3.[321]?XXX:5}', seq_task_i)
stim_presets['darpa_instr_seq_task_5'] = \
    ('%I1+I2+I5+I7+I3%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX' +
     'MP5.[123]?XXXMP7.[123]K[3]?XMP3.[456]?XXX:5}' +
     '%I3+I4+I6+I8+I2%{MP2.[0][1]?XXMP8.?1X?2X' +
     'MP6.[21][1][24][4][26][6][28]?XXMP4.[0]?XMP3.[321]?XXX:5}', seq_task_i)
stim_presets['darpa_instr_seq_task_6'] = \
    ('%I1+I2+I5+I7+I3+I4%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX' +
     'MP5.[123]?XXXMP7.[123]K[3]?XMP3.[456]?XXXMP4.[5]?X:5}' +
     '%I3+I4+I6+I8+I2+I5%{MP5.[456]?XXXMP2.[0][1]?XXMP8.?1X?2X' +
     'MP6.[21][1][24][4][26][6][28]?XXMP4.[0]?XMP3.[321]?XXX:5}', seq_task_i)

stim_presets['darpa_instr_stim_resp_demo1'] = \
    ('%I1+I2%A9?4X?9X%I1+I2+I3%A9?5XXX',
     'I1: VIS*FOR, DATA*POS1*TWO; I2: VIS*NIN, DATA*POS1*THR;' +
     'I3: VIS*FIV, DATA*(POS1*FOR + POS2*TWO + POS3*THR)')
stim_presets['darpa_instr_stim_resp_demo2'] = \
    ('%I1+I2%A9?4X?9X%I3+I4%A9?4X?9X',
     'I1: VIS*FOR, DATA*POS1*TWO; I2: VIS*NIN, DATA*POS1*THR;' +
     'I3: VIS*FOR, DATA*POS1*ONE; I4: VIS*NIN, DATA*POS1*EIG')
stim_presets['darpa_instr_stim_task_demo1'] = \
    ('%I1+I4%M1[#2]?XM2[427]?XXX',
     'I1: VIS*ONE, TASK*W; I2: VIS*TWO, TASK*R;' +
     'I3: VIS*ONE, TASK*M + DEC*FWD; I4: VIS*TWO, TASK*M + DEC*REV')
stim_presets['darpa_instr_stim_task_demo2'] = \
    ('%I1+I2%M1[<3725>]?XM2[<3725>]?X%I3+I4%M2[427]?XXX',
     'I1: VIS*ONE, TASK*W; I2: VIS*TWO, TASK*R;' +
     'I3: VIS*ONE, TASK*M + DEC*FWD; I4: VIS*TWO, TASK*M + DEC*REV')
stim_presets['darpa_instr_seq_task_demo'] = \
    ('%I1+I2+I3%MP3[<3725>]?XMP1[427]?XXXV[<3725>]?X',
     'I1: POS3, TASK*W; I2: POS2, TASK*R;I3: POS1, TASK*M + DEC*FWD')

# def_seq = 'A1[1]?XXA1[22]?XX'
# def_seq = '{A1[R]?X:5}'
# def_seq = '{A3[{R:7}]?{X:8}:5}'
# def_seq = '{A3[{R:7}]?{X:8}:160}'
# def_seq = 'A3[{R:7}]?{X:8}'
# def_seq = '%I1+I2%MP1.5[123]?XXXMP1.8[123]?XXX'
# def_i = 'I1: 0.5*POS1 + 0.5*VIS*FOR, TASK*M + DEC*FWD;' + \
#         'I2: 0.5*POS1 + 0.5*VIS*EIG, TASK*M + DEC*REV'

# Darpa instruction following + imagenet + adaptive motor presets
stim_presets['darpa_combined1'] = \
    ('{A3[#4#2#7#5]?XXXX:8}',
     'I1: VIS*GUENON, DATA*POS*FIV; I2: VIS*')

# ----- Configuration presets -----
cfg_presets = {}
cfg_presets['mtr_adapt_qvelff'] = ["mtr_dyn_adaptation=True",
                                   "mtr_forcefield='QVelForcefield'"]
cfg_presets['mtr_adapt_constff'] = ["mtr_dyn_adaptation=True",
                                    "mtr_forcefield='ConstForcefield'"]

cfg_presets['vis_imagenet'] = ["stim_module='imagenet'",
                               "vis_module='lif_imagenet'"]
cfg_presets['vis_imagenet_wta'] = ["stim_module='imagenet'",
                                   "vis_module='lif_imagenet_wta'"]

# Darpa adaptive motor demo configs
cfg_presets['darpa_adapt_qvelff_demo'] = \
    ["mtr_dyn_adaptation=True", "mtr_forcefield='QVelForcefield'",
     "probe_graph_config='ProbeCfgDarpaMotor'"]
cfg_presets['darpa_adapt_constff_demo'] = \
    ["mtr_dyn_adaptation=True", "mtr_forcefield='ConstForcefield'",
     "probe_graph_config='ProbeCfgDarpaMotor'"]

# Darpa imagenet demo configs
cfg_presets['darpa_vis_imagenet'] = \
    ["stim_module='imagenet'", "vis_module='lif_imagenet'",
     "probe_graph_config='ProbeCfgDarpaVisionImagenet'"]
cfg_presets['darpa_vis_imagenet_wta'] = \
    ["stim_module='imagenet'", "vis_module='lif_imagenet_wta'",
     "probe_graph_config='ProbeCfgDarpaVisionImagenet'"]

# ----- Definite maximum probe time (if est_sim_time > max_probe_time,
#       disable probing)
max_probe_time = 60

# ----- Add current directory to system path ---
cur_dir = os.getcwd()

# ----- Parse arguments -----
parser = argparse.ArgumentParser(description='Script for running Spaun.')
parser.add_argument(
    '-d', type=int, default=def_dim,
    help='Number of dimensions to use for the semantic pointers.')
parser.add_argument(
    '-t', type=float, default=-1,
    help=('Simulation run time in seconds. If undefined, will be estimated' +
          ' from the stimulus sequence.'))
parser.add_argument(
    '-n', type=int, default=1,
    help='Number of batches to run (each batch is a new model).')

parser.add_argument(
    '-s', type=str, default=def_seq,
    help='Stimulus sequence. Use digits to use canonical digits, prepend a ' +
         '"#" to a digit to use handwritten digits, a "[" for the open ' +
         'bracket, a "]" for the close bracket, and a "X" for each expected ' +
         'motor response. e.g. A3[1234]?XXXX or A0[#1]?X')
# Stimulus formats:
# Special characters - A [ ] ?
# To denote Spaun stereotypical numbers: 0 1 2 3 4 5 6 7 8 9
# To denote spaces for possible answers: X
# To denote specific image classes: #0 or #100, (either a # or non-digit will
#                                                partition numbers)
# To denote a image chosen using an array index: <1000>
# To denote random numbers chosen without replacement: N
# To denote random numbers chosen with replacement: R
# To denote 'reverse' option for memory task: B
# To denote matched random digits (with replacement): a - z (lowercase char)
# To denote forced blanks: .
# To denote changes in given instructions (see below): %INSTR_STR%
parser.add_argument(
    '-i', type=str, default=def_i,
    help='Instructions event sequence. Use the following format to provide ' +
         'customized instructions to spaun (which can then be put into the ' +
         'stimulus string using %%INSTR_KEYN+INSTR_KEYM%%": ' +
         '"INSTR_KEY: ANTECEDENT_SP_STR, CONSEQUENCE_SP_STR; ..."' +
         'e.g. "I1: TASK*INSTR + VIS*ONE, TRFM*POS1*THR", and the stimulus ' +
         'string: "%%I1+I2%%A0[0]?XX"')
# Note: For sequential position instructions, instruction must be encoded with
#       POS sp. E.g. I1: POS1+VIS*ONE, TASK*C
parser.add_argument(
    '--stim_preset', type=str, default='',
    help='Stimulus (stimulus sequence and instruction sequence pairing) to ' +
         'use for Spaun stimulus. Overrides -s and -i command line options ' +
         'if they are provided.')

parser.add_argument(
    '-b', type=str, default='ref',
    help='Backend to use for Spaun. One of ["ref", "ocl", "mpi", "spinn"]')
parser.add_argument(
    '--data_dir', type=str, default=os.path.join(cur_dir, 'data'),
    help='Directory to store output data.')
parser.add_argument(
    '--noprobes', action='store_true',
    help='Supply to disable probes.')
parser.add_argument(
    '--probeio', action='store_true',
    help='Supply to generate probe data for spaun inputs and outputs.' +
         '(recorded in a separate probe data file)')
parser.add_argument(
    '--seed', type=int, default=-1,
    help='Random seed to use.')
parser.add_argument(
    '--showgrph', action='store_true',
    help='Supply to show graphing of probe data.')
parser.add_argument(
    '--showanim', action='store_true',
    help='Supply to show animation of probe data.')
parser.add_argument(
    '--showiofig', action='store_true',
    help='Supply to show Spaun input/output figure.')
parser.add_argument(
    '--tag', type=str, default="",
    help='Tag string to apply to probe data file name.')
parser.add_argument(
    '--enable_cache', action='store_true',
    help='Supply to use nengo caching system when building the nengo model.')

parser.add_argument(
    '--ocl', action='store_true',
    help='Supply to use the OpenCL backend (will override -b).')
parser.add_argument(
    '--ocl_platform', type=int, default=-1,
    help=('OCL Only: List index of the OpenCL platform to use. OpenCL ' +
          ' backend can be listed using "pyopencl.get_platforms()"'))
parser.add_argument(
    '--ocl_device', type=int, default=-1,
    help=('OCL Only: List index of the device on the OpenCL platform to use.' +
          ' OpenCL devices can be listed using ' +
          '"pyopencl.get_platforms()[X].get_devices()" where X is the index ' +
          'of the plaform to use.'))
parser.add_argument(
    '--ocl_profile', action='store_true',
    help='Supply to use NengoOCL profiler.')

parser.add_argument(
    '--mpi', action='store_true',
    help='Supply to use the MPI backend (will override -b).')
parser.add_argument(
    '--mpi_save', type=str, default='spaun.net',
    help=('MPI Only: Filename to use to write the generated Spaun network ' +
          'to. Defaults to "spaun.net". *Note: Final filename includes ' +
          'neuron type, dimensionality, and stimulus information.'))
parser.add_argument(
    '--mpi_p', type=int, default=def_mpi_p,
    help='MPI Only: Number of processors to use.')
parser.add_argument(
    '--mpi_p_auto', action='store_true',
    help='MPI Only: Use the automatic partitioner')
parser.add_argument(
    '--mpi_compress_save', action='store_true',
    help='Supply to compress the saved net file with gzip.')

parser.add_argument(
    '--spinn', action='store_true',
    help='Supply to use the SpiNNaker backend (will override -b).')

parser.add_argument(
    '--nengo_gui', action='store_true',
    help='Supply to use the nengo_viz vizualizer to run Spaun.')

parser.add_argument(
    '--config', type=str, nargs='*',
    help="Use to set the various parameters in Spaun's configuration. Takes" +
         " arguments in list format. Each argument should be in the format" +
         " ARG_NAME=ARG_VALUE. " +
         "\nE.g. --config sim_dt=0.002 mb_gate_scale=0.8 " +
         "\"raw_seq_str='A1[123]?XX'\"" +
         "\nNOTE: Will override all other options that set configuration" +
         " options (i.e. --seed, --d, --s)" +
         '\nNOTE: Use quotes (") to encapsulate strings if you encounter' +
         ' problems.')
parser.add_argument(
    '--config_presets', type=str, nargs='*',
    help="Use to provide preset configuration options (which can be " +
         "individually provided using --config). Appends to list of " +
         "configuration options provided through --config.")

parser.add_argument(
    '--debug', action='store_true',
    help='Supply to output debug stuff.')

args = parser.parse_args()

# ----- Nengo RC Cache settings -----
# Disable cache unless seed is set (i.e. seed > 0) or if the '--enable_cache'
# option is given
if args.seed > 0 or args.enable_cache:
    print "USING CACHE"
    nengo.rc.set("decoder_cache", "enabled", "True")
else:
    print "NOT USING CACHE"
    nengo.rc.set("decoder_cache", "enabled", "False")

# ----- Backend Configurations -----
cfg.backend = args.b
if args.ocl:
    cfg.backend = 'ocl'
if args.mpi:
    cfg.backend = 'mpi'
if args.spinn:
    cfg.backend = 'spinn'

print "BACKEND: %s" % cfg.backend.upper()

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
    print ("\n======================== RUN %i OF %i ========================" %
           (n + 1, args.n))

    # ----- Seeeeeeeed -----
    if args.seed < 0:
        seed = int(time.time())
    else:
        seed = args.seed

    cfg.set_seed(seed)
    print "MODEL SEED: %i" % cfg.seed

    # ----- Model Configurations -----
    vocab.sp_dim = args.d
    cfg.data_dir = args.data_dir

    # Parse --config options
    if len(config_list) > 0:
        print "USING CONFIGURATION OPTIONS: "
        for cfg_options in config_list:
            cfg_opts = cfg_options.split('=')
            cfg_param = cfg_opts[0]
            cfg_value = cfg_opts[1]
            if hasattr(cfg, cfg_param):
                print "  * cfg: " + str(cfg_options)
                setattr(cfg, cfg_param, eval(cfg_value))
            elif hasattr(experiment, cfg_param):
                print "  * experiment: " + str(cfg_options)
                setattr(experiment, cfg_param, eval(cfg_value))
            elif hasattr(vocab, cfg_param):
                print "  * vocab: " + str(cfg_options)
                setattr(vocab, cfg_param, eval(cfg_value))

    # ----- Check if data folder exists -----
    if not(os.path.isdir(cfg.data_dir) and os.path.exists(cfg.data_dir)):
        raise RuntimeError('Data directory "%s"' % (cfg.data_dir) +
                           ' does not exist. Please ensure the correct path' +
                           ' has been specified.')

    # ----- Spaun imports -----
    from _spaun.utils import get_total_n_neurons
    from _spaun.spaun_main import Spaun

    from _spaun.modules.stim import stim_data
    from _spaun.modules.vision import vis_data
    from _spaun.modules.motor import mtr_data

    # ----- Enable debug logging -----
    if args.debug:
        nengo.log('debug')

    # ----- Experiment and vocabulary initialization -----
    experiment.initialize(stim_seq_str, stim_data.get_image_ind,
                          stim_data.get_image_label,
                          cfg.mtr_est_digit_response_time, instr_seq_str,
                          cfg.rng)
    vocab.initialize(stim_data.stim_SP_labels, experiment.num_learn_actions,
                     cfg.rng)
    vocab.initialize_mtr_vocab(mtr_data.dimensions, mtr_data.sps)
    vocab.initialize_vis_vocab(vis_data.dimensions, vis_data.sps)

    # ----- Configure output log files -----
    if cfg.use_mpi:
        sys.path.append('C:\\Users\\xchoo\\GitHub\\nengo_mpi')

        mpi_save = args.mpi_save.split('.')
        mpi_savename = '.'.join(mpi_save[:-1])
        mpi_saveext = mpi_save[-1]

        cfg.probe_data_filename = get_probe_data_filename(mpi_savename,
                                                          suffix=args.tag)
    else:
        cfg.probe_data_filename = get_probe_data_filename(suffix=args.tag)

    # ----- Initalize looger and write header data -----
    logger.initialize(cfg.data_dir, cfg.probe_data_filename[:-4] + '_log.txt')
    cfg.write_header()
    experiment.write_header()
    vocab.write_header()
    logger.flush()

    # ----- Raw stimulus seq -----
    print "RAW STIM SEQ: %s" % (str(experiment.raw_seq_str))

    # ----- Spaun proper -----
    model = Spaun()

    # ----- Display stimulus seq -----
    print "PROCESSED RAW STIM SEQ: %s" % (str(experiment.raw_seq_list))
    print "STIMULUS SEQ: %s" % (str(experiment.stim_seq_list))

    # ----- Calculate runtime -----
    # Note: Moved up here so that we have data to disable probes if necessary
    runtime = args.t if args.t > 0 else experiment.get_est_simtime()

    # ----- Set up probes -----
    from _spaun import probes as probe_module

    make_probes = not args.noprobes
    if runtime > max_probe_time and make_probes:
        print (">>> !!! WARNING !!! EST RUNTIME > %0.2fs - DISABLING PROBES" %
               max_probe_time)
        make_probes = False

    if make_probes:
        print "PROBE FILENAME: %s" % cfg.probe_data_filename
        default_probe_config = getattr(probe_module, cfg.probe_graph_config)
        probe_cfg = default_probe_config(model, vocab, cfg.sim_dt,
                                         cfg.data_dir,
                                         cfg.probe_data_filename)

    # ----- Set up animation probes -----
    if args.showanim or args.showiofig or args.probeio:
        anim_probe_data_filename = cfg.probe_data_filename[:-4] + '_anim.npz'
        default_anim_config = getattr(probe_module, cfg.probe_anim_config)
        print "ANIM PROBE FILENAME: %s" % anim_probe_data_filename
        probe_anim_cfg = default_anim_config(model, vocab,
                                             cfg.sim_dt, cfg.data_dir,
                                             anim_probe_data_filename)

    # ----- Neuron count debug -----
    print "MODEL N_NEURONS:  %i" % (get_total_n_neurons(model))
    if hasattr(model, 'vis'):
        print "- vis   n_neurons: %i" % (get_total_n_neurons(model.vis))
    if hasattr(model, 'ps'):
        print "- ps    n_neurons: %i" % (get_total_n_neurons(model.ps))
    if hasattr(model, 'bg'):
        print "- bg    n_neurons: %i" % (get_total_n_neurons(model.bg))
    if hasattr(model, 'thal'):
        print "- thal  n_neurons: %i" % (get_total_n_neurons(model.thal))
    if hasattr(model, 'enc'):
        print "- enc   n_neurons: %i" % (get_total_n_neurons(model.enc))
    if hasattr(model, 'mem'):
        print "- mem   n_neurons: %i" % (get_total_n_neurons(model.mem))
    if hasattr(model, 'trfm'):
        print "- trfm  n_neurons: %i" % (get_total_n_neurons(model.trfm))
    if hasattr(model, 'instr'):
        print "- instr n_neurons: %i" % (get_total_n_neurons(model.instr))
    if hasattr(model, 'dec'):
        print "- dec   n_neurons: %i" % (get_total_n_neurons(model.dec))
    if hasattr(model, 'mtr'):
        print "- mtr   n_neurons: %i" % (get_total_n_neurons(model.mtr))

    # ----- Connections count debug -----
    print "MODEL N_CONNECTIONS: %i" % (len(model.all_connections))

    # ----- Spaun simulation build -----
    print "START BUILD"
    timestamp = time.time()

    if args.nengo_gui:
        # Set environment variables (for nengo_gui)
        if cfg.use_opencl:
            if args.ocl_platform >= 0 and args.ocl_device >= 0:
                os.environ['PYOPENCL_CTX'] = '%s:%s' % (args.ocl_platform,
                                                        args.ocl_device)
            else:
                raise RuntimeError('Error - OCL platform and device must be' +
                                   'specified when using ocl with nengo_gui.' +
                                   ' Use the --ocl_platform and --ocl_device' +
                                   ' argument options to set.')

        print "STARTING NENGO_GUI"
        import nengo_gui
        nengo_gui.GUI(__file__, model=model, locals=locals(),
                      editor=False).start()
        print "NENGO_GUI STOPPED"
        sys.exit()

    if cfg.use_opencl:
        import pyopencl as cl
        import nengo_ocl

        print "------ OCL ------"
        print "AVAILABLE PLATFORMS:"
        print '  ' + '\n  '.join(map(str, cl.get_platforms()))

        if args.ocl_platform >= 0:
            pltf = cl.get_platforms()[args.ocl_platform]
            print "USING PLATFORM:"
            print '  ' + str(pltf)

            print "AVAILABLE DEVICES:"
            print '  ' + '\n  '.join(map(str, pltf.get_devices()))
            if args.ocl_device >= 0:
                ctx = cl.Context([pltf.get_devices()[args.ocl_device]])
                print "USING DEVICE:"
                print '  ' + str(pltf.get_devices()[args.ocl_device])
            else:
                ctx = cl.Context(pltf.get_devices())
                print "USING DEVICES:"
                print '  ' + '\n  '.join(map(str, pltf.get_devices()))
            sim = nengo_ocl.Simulator(model, dt=cfg.sim_dt, context=ctx,
                                      profiling=args.ocl_profile)
        else:
            sim = nengo_ocl.Simulator(model, dt=cfg.sim_dt,
                                      profiling=args.ocl_profile)
    elif cfg.use_mpi:
        import nengo_mpi

        mpi_savefile = \
            ('+'.join([cfg.get_probe_data_filename(mpi_savename)[:-4],
                      ('%ip' % args.mpi_p if not args.mpi_p_auto else 'autop'),
                      '%0.2fs' % experiment.get_est_simtime()]) + '.' +
             mpi_saveext)
        mpi_savefile = os.path.join(cfg.data_dir, mpi_savefile)

        print "USING MPI - Saving to: %s" % (mpi_savefile)

        if args.mpi_p_auto:
            assignments = {}
            for n, module in enumerate(model.modules):
                assignments[module] = n
            sim = nengo_mpi.Simulator(model, dt=cfg.sim_dt,
                                      assignments=assignments,
                                      save_file=mpi_savefile)
        else:
            partitioner = nengo_mpi.Partitioner(args.mpi_p)
            sim = nengo_mpi.Simulator(model, dt=cfg.sim_dt,
                                      partitioner=partitioner,
                                      save_file=mpi_savefile)
    else:
        sim = nengo.Simulator(model, dt=cfg.sim_dt)

    t_build = time.time() - timestamp
    timestamp = time.time()
    print "BUILD FINISHED - build time: %fs" % t_build

    # ----- Spaun simulation run -----
    experiment.reset()
    if cfg.use_opencl or cfg.use_ref:
        print "START SIM - est_runtime: %f" % runtime
        sim.run(runtime)

        # Close output logging file
        logger.close()

        if args.ocl_profile:
            sim.print_plans()
            sim.print_profiling()

        t_simrun = time.time() - timestamp
        print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
        print "FINISHED! - Build time: %fs, Sim time: %fs" % (t_build,
                                                              t_simrun)
    else:
        print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
        print "FINISHED! - Build time: %fs" % (t_build)

        if args.mpi_compress_save:
            import gzip
            print "COMPRESSING net file to '%s'" % (mpi_savefile + '.gz')

            with open(mpi_savefile, 'rb') as f_in:
                with gzip.open(mpi_savefile + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)

            os.remove(mpi_savefile)

            print "UPLOAD '%s' to MPI cluster and decompress to run" % \
                (mpi_savefile + '.gz')
        else:
            print "UPLOAD '%s' to MPI cluster to run" % mpi_savefile
        t_simrun = -1

    # ----- Generate debug printouts -----
    n_bytes_ev = 0
    n_bytes_gain = 0
    n_bytes_bias = 0
    n_ens = 0
    for ens in sim.model.toplevel.all_ensembles:
        n_bytes_ev += sim.model.params[ens].eval_points.nbytes
        n_bytes_gain += sim.model.params[ens].gain.nbytes
        n_bytes_bias += sim.model.params[ens].bias.nbytes
        n_ens += 1

    if args.debug:
        print("## DEBUG: num bytes used for eval points: %s B" %
              ("{:,}".format(n_bytes_ev)))
        print("## DEBUG: num bytes used for gains: %s B" %
              ("{:,}".format(n_bytes_gain)))
        print("## DEBUG: num bytes used for biases: %s B" %
              ("{:,}".format(n_bytes_bias)))
        print("## DEBUG: num ensembles: %s" % n_ens)

    # ----- Close simulator -----
    if hasattr(sim, 'close'):
        sim.close()

    # ----- Write probe data to file -----
    if make_probes and not cfg.use_mpi:
        print "WRITING PROBE DATA TO FILE"
        probe_cfg.write_simdata_to_file(sim, experiment)

        # Assemble graphing subprocess call string
        subprocess_call_list = ["python",
                                os.path.join(cur_dir,
                                             'disp_probe_data.py'),
                                '"' + cfg.probe_data_filename + '"',
                                '--data_dir', '"' + cfg.data_dir + '"',
                                '--showgrph']

        # Log subprocess call
        logger.write("\n# " + " ".join(subprocess_call_list))

        if args.showgrph:
            # Open subprocess
            print "CALLING: \n%s" % (" ".join(subprocess_call_list))
            import subprocess
            subprocess.Popen(subprocess_call_list)

    if (args.showanim or args.showiofig or args.probeio) and not cfg.use_mpi:
        print "WRITING ANIMATION PROBE DATA TO FILE"
        probe_anim_cfg.write_simdata_to_file(sim, experiment)

        # Assemble graphing subprocess call string
        subprocess_call_list = ["python",
                                os.path.join(cur_dir,
                                             'disp_probe_data.py'),
                                '"' + anim_probe_data_filename + '"',
                                '--data_dir', '"' + cfg.data_dir + '"']
        if args.showanim:
            subprocess_call_list += ['--showanim']
        if args.showiofig:
            subprocess_call_list += ['--showiofig']

        # Log subprocess call
        logger.write("\n# " + " ".join(subprocess_call_list))

        if args.showanim or args.showiofig:
            # Open subprocess
            print "CALLING: \n%s" % (" ".join(subprocess_call_list))
            import subprocess
            subprocess.Popen(subprocess_call_list)

    # ----- Write runtime data -----
    runtime_filename = os.path.join(cfg.data_dir, 'runtimes.txt')
    rt_file = open(runtime_filename, 'a')
    rt_file.write('# ---------- TIMESTAMP: %i -----------\n' % timestamp)
    rt_file.write('Backend: %s | Num neurons: %i | Tag: %s | Seed: %i\n' %
                  (cfg.backend, get_total_n_neurons(model), args.tag,
                   cfg.seed))
    if args.config is not None:
        rt_file.write('Config options: %s\n' % (str(args.config)))
    rt_file.write('Build time: %fs | Model sim time: %fs | ' % (t_build,
                                                                runtime))
    rt_file.write('Sim wall time: %fs\n' % (t_simrun))
    rt_file.close()

    # ----- Cleanup -----
    model = None
    sim = None
    probe_data = None
