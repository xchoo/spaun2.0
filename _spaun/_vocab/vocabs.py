from copy import deepcopy as copy

from nengo.spa import Vocabulary

from ..config import cfg
from ..utils import sum_vocab_vecs


################# Semantic pointer (strings) definitions ######################
# --- Numerical semantic pointers ---
# num_sp_strs = ["ZER", "ONE", "TWO", "THR", "FOR",
#            "FIV", "SIX", "SEV", "EIG", "NIN"]
num_sp_strs = ["ZER", "ONE", "TWO", "THR", "FOR", "FIV"]    # For testing

# --- Task semantic pointer list ---
# W - Drawing (Copying visual input)
# R - Recognition
# L - Learning (Bandit Task)
# M - Memory (forward serial recall)
# C - Counting
# A - Answering
# V - Rapid Variable Creation
# F - Fluid Induction (Ravens)
# X - Task precursor
# DEC - Decoding task (output to motor system)
# DECW - Decoding task (output to motor system, but for drawing task)
task_sp_strs = ["W", "R", "L", "M", "C", "A", "V", "F", "X", "DEC", "DECW"]
task_vis_sp_strs = ["A"]

# --- Misc visual semantic pointers ---
misc_vis_sp_strs = ["OPEN", "CLOSE", "QM"]

# --- List of all visual semantic pointers ---
vis_sp_strs = copy(num_sp_strs)
vis_sp_strs.extend(task_vis_sp_strs)
vis_sp_strs.extend(misc_vis_sp_strs)

# --- Position (enumerated) semantic pointers ---
pos_sp_strs = ["POS%i" % (i + 1) for i in range(cfg.max_enum_list_pos)]

# --- Operations semantic pointers
ops_sp_strs = ["ADD", "INC"]

# --- Unitary semantic pointers
unitary_sp_strs = [num_sp_strs[0], pos_sp_strs[0]]
unitary_sp_strs.extend(ops_sp_strs)


######################### Vocabulary definitions ##############################
# --- Primary vocabulary ---
vocab = Vocabulary(cfg.sp_dim, unitary=unitary_sp_strs)

# --- Add numerical sp's ---
vocab.parse("%s+%s" % (ops_sp_strs[0], num_sp_strs[0]))
add_sp = vocab[ops_sp_strs[0]]
num_sp = vocab[num_sp_strs[0]]
for i in range(len(num_sp_strs) - 1):
    num_sp = num_sp * add_sp
    vocab.add(num_sp_strs[i + 1], num_sp)

# --- Add positional sp's ---
vocab.parse("%s+%s" % (ops_sp_strs[1], pos_sp_strs[0]))
inc_sp = vocab[ops_sp_strs[1]]
pos_sp = vocab[pos_sp_strs[0]]
for i in range(len(pos_sp_strs) - 1):
    pos_sp = pos_sp * inc_sp
    vocab.add(pos_sp_strs[i + 1], pos_sp)

# --- Add other visual sp's ---
vocab.parse("+".join(misc_vis_sp_strs))
vocab.parse("+".join(task_vis_sp_strs))

# --- Add task sp's ---
vocab.parse("+".join(task_sp_strs))

### --- Motor vocabulary (for debug purposes) ---
mtr_vocab = Vocabulary(cfg.mtr_dim)
mtr_vocab.parse("+".join(num_sp_strs))

####################### Sub-vocabulary definitions ############################
vis_vocab = vocab.create_subset(vis_sp_strs)
vis_vocab_nums_inds = range(len(num_sp_strs))
vis_vocab_syms_inds = range(len(num_sp_strs), len(vis_sp_strs))

pos_vocab = vocab.create_subset(pos_sp_strs)

item_vocab = vocab.create_subset(num_sp_strs)

task_vocab = vocab.create_subset(task_sp_strs)


################## Enumerated vocabulary definitions ##########################
# --- Enumerated vocabulary, enumerates all possible combinations of position
#     and item vectors (for debug purposes)
enum_vocab = Vocabulary(cfg.sp_dim)

for pos in pos_sp_strs:
    for num in num_sp_strs:
        enum_vocab.add("%s*%s" % (pos, num), vocab[pos] * vocab[num])


################ Semantic pointer lists for signal generation #################
strs_to_inds = lambda l, ref: [ref.index(i) for i in l]

item_mb_gate_sp_strs = copy(num_sp_strs)
item_mb_gate_sp_inds = strs_to_inds(item_mb_gate_sp_strs, vis_sp_strs)
item_mb_rst_sp_strs = ["A", "OPEN"]
item_mb_rst_sp_inds = strs_to_inds(item_mb_rst_sp_strs, vis_sp_strs)

pos_mb_gate_sp_strs = copy(num_sp_strs)
pos_mb_gate_sp_strs.extend(["A", "OPEN", "QM"])
pos_mb_gate_sp_inds = strs_to_inds(pos_mb_gate_sp_strs, vis_sp_strs)
pos_mb_rst_sp_strs = ["A", "OPEN", "QM"]
pos_mb_rst_sp_inds = strs_to_inds(pos_mb_rst_sp_strs, vis_sp_strs)

task_mb_gate_sp_strs = copy(num_sp_strs)
task_mb_gate_sp_strs.extend(["A", "QM"])
task_mb_gate_sp_inds = strs_to_inds(task_mb_gate_sp_strs, vis_sp_strs)

dec_out_sel_sp_strs = ["DECW"]
dec_out_sel_sp_vecs = sum_vocab_vecs(vocab, dec_out_sel_sp_strs)
