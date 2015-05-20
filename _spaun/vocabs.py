from copy import deepcopy as copy

from nengo.spa import Vocabulary
# from ._spa import Vocabulary

from .config import cfg
from .utils import sum_vocab_vecs
from .utils import strs_to_inds


# ############### Semantic pointer (strings) definitions ######################
# --- Numerical semantic pointers ---
num_sp_strs = ['ZER', 'ONE', 'TWO', 'THR', 'FOR',
               'FIV', 'SIX', 'SEV', 'EIG', 'NIN']

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
# DECI - Decoding task (output to motor system, but for induction tasks)
ps_task_sp_strs = ['W', 'R', 'L', 'M', 'C', 'A', 'V', 'F', 'X', 'DEC', 'DECW',
                   'DECI']
ps_task_vis_sp_strs = ['A', 'C', 'F', 'K', 'L', 'M', 'P', 'R', 'V', 'W']

# --- Production system semantic pointers ---
ps_state_sp_strs = ['QAP', 'QAK', 'TRANS0', 'TRANS1', 'TRANS2', 'CNT']
ps_dec_sp_strs = ['FWD', 'REV', 'CNT']

# --- Misc visual semantic pointers ---
misc_vis_sp_strs = ['OPEN', 'CLOSE', 'SPACE', 'QM']

# --- 'I don't know' motor response vector
mtr_sp_strs = ['UNK']

# --- List of all visual semantic pointers ---
vis_sp_strs = copy(num_sp_strs)
vis_sp_strs.extend(misc_vis_sp_strs)
vis_sp_strs.extend(ps_task_vis_sp_strs)

# --- Position (enumerated) semantic pointers ---
pos_sp_strs = ['POS%i' % (i + 1) for i in range(cfg.max_enum_list_pos)]

# --- Operations semantic pointers
ops_sp_strs = ['ADD', 'INC']

# --- Unitary semantic pointers
unitary_sp_strs = [num_sp_strs[0], pos_sp_strs[0]]
unitary_sp_strs.extend(ops_sp_strs)


# ####################### Vocabulary definitions ##############################
# --- Primary vocabulary ---
vocab = Vocabulary(cfg.sp_dim, unitary=unitary_sp_strs, rng=cfg.rng)

# --- Add numerical sp's ---
vocab.parse('%s+%s' % (ops_sp_strs[0], num_sp_strs[0]))
add_sp = vocab[ops_sp_strs[0]]
num_sp = vocab[num_sp_strs[0]].copy()
for i in range(len(num_sp_strs) - 1):
    num_sp = num_sp.copy() * add_sp
    vocab.add(num_sp_strs[i + 1], num_sp)

# --- Add positional sp's ---
vocab.parse('%s+%s' % (ops_sp_strs[1], pos_sp_strs[0]))
inc_sp = vocab[ops_sp_strs[1]]
pos_sp = vocab[pos_sp_strs[0]].copy()
for i in range(len(pos_sp_strs) - 1):
    pos_sp = pos_sp.copy() * inc_sp
    vocab.add(pos_sp_strs[i + 1], pos_sp)

# --- Add other visual sp's ---
vocab.parse('+'.join(misc_vis_sp_strs))
vocab.parse('+'.join(ps_task_vis_sp_strs))

# --- Add production system sp's ---
vocab.parse('+'.join(ps_task_sp_strs))
vocab.parse('+'.join(ps_state_sp_strs))
vocab.parse('+'.join(ps_dec_sp_strs))

# ## --- Motor vocabularies (for debug purposes) ---
mtr_vocab = Vocabulary(cfg.mtr_dim, rng=cfg.rng)
mtr_vocab.parse('+'.join(num_sp_strs))

mtr_unk_vocab = Vocabulary(cfg.mtr_dim, rng=cfg.rng)
mtr_unk_vocab.parse(mtr_sp_strs[0])

mtr_disp_vocab = mtr_vocab.create_subset(num_sp_strs)
mtr_disp_vocab.add(mtr_sp_strs[0], mtr_unk_vocab[mtr_sp_strs[0]].v)

# ##################### Sub-vocabulary definitions ############################
vis_vocab = vocab.create_subset(vis_sp_strs)
vis_vocab_nums_inds = range(len(num_sp_strs))
vis_vocab_syms_inds = range(len(num_sp_strs), len(vis_sp_strs))

pos_vocab = vocab.create_subset(pos_sp_strs)

item_vocab = vocab.create_subset(num_sp_strs)

ps_task_vocab = vocab.create_subset(ps_task_sp_strs)
ps_state_vocab = vocab.create_subset(ps_state_sp_strs)
ps_dec_vocab = vocab.create_subset(ps_dec_sp_strs)

# ################ Enumerated vocabulary definitions ##########################
# --- Enumerated vocabulary, enumerates all possible combinations of position
#     and item vectors (for debug purposes)
enum_vocab = Vocabulary(cfg.sp_dim, rng=cfg.rng)
for pos in pos_sp_strs:
    for num in num_sp_strs:
        enum_vocab.add('%s*%s' % (pos, num), vocab[pos] * vocab[num])

pos1_vocab = Vocabulary(cfg.sp_dim, rng=cfg.rng)
for num in num_sp_strs:
    pos1_vocab.add('%s*%s' % (pos_sp_strs[0], num),
                   vocab[pos_sp_strs[0]] * vocab[num])

# ############## Semantic pointer lists for signal generation #################
item_mb_gate_sp_strs = copy(num_sp_strs)
item_mb_gate_sp_inds = strs_to_inds(item_mb_gate_sp_strs, vis_sp_strs)
item_mb_rst_sp_strs = ['A', 'OPEN']
item_mb_rst_sp_inds = strs_to_inds(item_mb_rst_sp_strs, vis_sp_strs)

ave_mb_gate_sp_strs = ['CLOSE']
ave_mb_gate_sp_inds = strs_to_inds(ave_mb_gate_sp_strs, vis_sp_strs)
ave_mb_rst_sp_strs = ['A']
ave_mb_rst_sp_inds = strs_to_inds(ave_mb_rst_sp_strs, vis_sp_strs)

pos_mb_gate_sp_strs = copy(num_sp_strs)
# pos_mb_gate_sp_strs.extend(['A', 'OPEN', 'QM'])
pos_mb_gate_sp_inds = strs_to_inds(pos_mb_gate_sp_strs, vis_sp_strs)
pos_mb_rst_sp_strs = ['A', 'OPEN', 'QM']
pos_mb_rst_sp_inds = strs_to_inds(pos_mb_rst_sp_strs, vis_sp_strs)

ps_task_mb_gate_sp_strs = copy(num_sp_strs)
ps_task_mb_gate_sp_strs.extend(['QM'])
ps_task_mb_gate_sp_inds = strs_to_inds(ps_task_mb_gate_sp_strs, vis_sp_strs)

ps_task_mb_rst_sp_strs = ['A']
ps_task_mb_rst_sp_inds = strs_to_inds(ps_task_mb_rst_sp_strs, vis_sp_strs)

ps_state_mb_gate_sp_strs = ['CLOSE', 'K', 'P']
ps_state_mb_gate_sp_inds = strs_to_inds(ps_state_mb_gate_sp_strs, vis_sp_strs)

ps_state_mb_rst_sp_strs = ['A']
ps_state_mb_rst_sp_inds = strs_to_inds(ps_state_mb_rst_sp_strs, vis_sp_strs)

ps_dec_mb_gate_sp_strs = ['F', 'R', 'QM']
ps_dec_mb_gate_sp_inds = strs_to_inds(ps_dec_mb_gate_sp_strs, vis_sp_strs)

ps_dec_mb_rst_sp_strs = ['A']
ps_dec_mb_rst_sp_inds = strs_to_inds(ps_dec_mb_rst_sp_strs, vis_sp_strs)

# Note: sum_vocab_vecs have to be fed through threshold before use.
dec_out_sel_sp_strs = ['DECW']
dec_out_sel_sp_vecs = sum_vocab_vecs(vocab, dec_out_sel_sp_strs)

dec_pos_gate_sp_strs = ['DECW', 'DEC', 'DECC']
dec_pos_gate_sp_vecs = sum_vocab_vecs(vocab, dec_pos_gate_sp_strs)
