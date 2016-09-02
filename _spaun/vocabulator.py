import time
import numpy as np

from nengo.spa import Vocabulary
from nengo.spa import SemanticPointer

from loggerator import logger


class SpaunVocabulary(object):
    def __init__(self):
        self.main = None

        self.sp_dim = 512
        self.mtr_dim = 50
        self.vis_dim = 200

        # ############ Semantic pointer (strings) definitions #################
        # --- Numerical semantic pointers ---
        self.num_sp_strs = ['ZER', 'ONE', 'TWO', 'THR', 'FOR',
                            'FIV', 'SIX', 'SEV', 'EIG', 'NIN']
        self.n_num_sp = len(self.num_sp_strs)

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
        self.ps_task_sp_strs = ['W', 'R', 'L', 'M', 'C', 'A', 'V', 'F', 'X',
                                'DEC', 'REACT', 'INSTR']
        self.ps_task_vis_sp_strs = ['A', 'C', 'F', 'K', 'L', 'M', 'P', 'R',
                                    'V', 'W']
        # --- Task visual semantic pointer usage ---
        # A - Task initialization
        # F - Forward recall
        # R - Reverse recall
        # K - Q&A 'kind' probe
        # P - Q&A 'position' probe

        # --- Production system semantic pointers ---
        # DECW - Decoding state (output to motor system, but for drawing task)
        # DECI - Decoding state (output to motor system, but for inductn tasks)
        self.ps_state_sp_strs = ['QAP', 'QAK', 'TRANS0', 'TRANS1', 'TRANS2',
                                 'CNT0', 'CNT1', 'LEARN', 'DIRECT', 'INSTRP',
                                 'INSTRV']
        self.ps_dec_sp_strs = ['FWD', 'REV', 'CNT', 'DECW', 'DECI', 'NONE']

        # --- Misc actions semantic pointers
        self.ps_action_sp_strs = None
        self.min_num_ps_actions = 3

        # --- Misc visual semantic pointers ---
        self.misc_vis_sp_strs = ['OPEN', 'CLOSE', 'SPACE', 'QM']

        # --- Misc state semantic pointers ---
        self.misc_ps_sp_strs = ['MATCH', 'NO_MATCH']

        # --- 'I don't know' motor response vector
        self.mtr_sp_strs = ['UNK']

        # --- List of all visual semantic pointers ---
        self.vis_sp_strs = list(self.num_sp_strs)
        self.vis_sp_strs.extend(self.misc_vis_sp_strs)
        self.vis_sp_strs.extend(self.ps_task_vis_sp_strs)

        # --- Position (enumerated) semantic pointers ---
        self.pos_sp_strs = None
        self.max_enum_list_pos = 8

        # --- Operations semantic pointers
        self.ops_sp_strs = ['ADD', 'INC']

        # --- Reward semantic pointers
        self.reward_n_sp_str = self.num_sp_strs[0]
        self.reward_y_sp_str = self.num_sp_strs[1]
        self.reward_sp_strs = [self.reward_n_sp_str, self.reward_y_sp_str]

        # --- Instruction processing input and output tags
        self.instr_tag_strs = ['VIS', 'TASK', 'STATE', 'DEC', 'DATA', 'ENABLE']

    def write_header(self):
        logger.write('# Spaun Vocabulary Options:\n')
        logger.write('# -------------------------\n')
        for param_name in sorted(self.__dict__.keys()):
            param_value = getattr(self, param_name)
            if not callable(param_value) and not isinstance(param_value, list)\
               and not isinstance(param_value, Vocabulary) \
               and not isinstance(param_value, SemanticPointer) \
               and not isinstance(param_value, np.ndarray):
                logger.write('# - %s = %s\n' % (param_name, param_value))
        logger.write('\n')

    def initialize(self, num_learn_actions=3, rng=0):
        if rng == 0:
            rng = np.random.RandomState(int(time.time()))

        # ############### Semantic pointer list definitions ###################
        # --- Position (enumerated) semantic pointers ---
        self.pos_sp_strs = ['POS%i' % (i + 1)
                            for i in range(self.max_enum_list_pos)]

        # --- Unitary semantic pointers
        self.unitary_sp_strs = [self.num_sp_strs[0], self.pos_sp_strs[0]]
        self.unitary_sp_strs.extend(self.ops_sp_strs)

        # --- Production system (action) semantic pointers ---
        self.ps_action_learn_sp_strs = ['A%d' % (i + 1) for i in
                                        range(num_learn_actions)]
        self.ps_action_misc_sp_strs = []
        self.ps_action_sp_strs = (self.ps_action_learn_sp_strs +
                                  self.ps_action_misc_sp_strs)

        # #################### Vocabulary definitions #########################
        # --- Primary vocabulary ---
        self.main = Vocabulary(self.sp_dim, unitary=self.unitary_sp_strs,
                               rng=rng)

        # --- Add numerical sp's ---
        self.main.parse('%s+%s' % (self.ops_sp_strs[0], self.num_sp_strs[0]))
        add_sp = self.main[self.ops_sp_strs[0]]
        num_sp = self.main[self.num_sp_strs[0]].copy()
        for i in range(len(self.num_sp_strs) - 1):
            num_sp = num_sp.copy() * add_sp
            self.main.add(self.num_sp_strs[i + 1], num_sp)

        self.add_sp = add_sp

        # --- Add positional sp's ---
        self.main.parse('%s+%s' % (self.ops_sp_strs[1], self.pos_sp_strs[0]))
        inc_sp = self.main[self.ops_sp_strs[1]]
        pos_sp = self.main[self.pos_sp_strs[0]].copy()
        for i in range(len(self.pos_sp_strs) - 1):
            pos_sp = pos_sp.copy() * inc_sp
            self.main.add(self.pos_sp_strs[i + 1], pos_sp)

        self.inc_sp = inc_sp

        # --- Add other visual sp's ---
        self.main.parse('+'.join(self.misc_vis_sp_strs))
        self.main.parse('+'.join(self.ps_task_vis_sp_strs))

        # --- Add production system sp's ---
        self.main.parse('+'.join(self.ps_task_sp_strs))
        self.main.parse('+'.join(self.ps_state_sp_strs))
        self.main.parse('+'.join(self.ps_dec_sp_strs))
        if len(self.ps_action_sp_strs) > 0:
            self.main.parse('+'.join(self.ps_action_sp_strs))
        self.main.parse('+'.join(self.misc_ps_sp_strs))

        # --- Add instruction processing system sp's ---
        self.main.parse('+'.join(self.instr_tag_strs))

        # ################# Sub-vocabulary definitions ########################
        self.vis_main = self.main.create_subset(self.vis_sp_strs)

        self.pos = self.main.create_subset(self.pos_sp_strs)

        self.item = self.main.create_subset(self.num_sp_strs)
        self.item_1_index = self.main.create_subset(self.num_sp_strs[1:])

        self.ps_task = self.main.create_subset(self.ps_task_sp_strs)
        self.ps_state = self.main.create_subset(self.ps_state_sp_strs)
        self.ps_dec = self.main.create_subset(self.ps_dec_sp_strs)
        self.ps_cmp = self.main.create_subset(self.misc_ps_sp_strs)
        self.ps_action = self.main.create_subset(self.ps_action_sp_strs)
        self.ps_action_learn = \
            self.main.create_subset(self.ps_action_learn_sp_strs)

        self.reward = self.main.create_subset(self.reward_sp_strs)

        self.instr = self.main.create_subset(self.instr_tag_strs)

        # ############ Enumerated vocabulary definitions ######################
        # --- Enumerated vocabulary, enumerates all possible combinations of
        #     position and item vectors (for debug purposes)
        self.enum = Vocabulary(self.sp_dim, rng=rng)
        for pos in self.pos_sp_strs:
            for num in self.num_sp_strs:
                sp_str = '%s*%s' % (pos, num)
                self.enum.add(sp_str, self.main.parse(sp_str))

        self.pos1 = Vocabulary(self.sp_dim, rng=rng)
        for num in self.num_sp_strs:
            sp_str = '%s*%s' % (self.pos_sp_strs[0], num)
            self.pos1.add(sp_str, self.main.parse(sp_str))

        # ############ Instruction vocabulary definitions #####################
        # --- ANTECEDENT and CONSEQUENCE permutation transforms
        self.perm_ant = np.arange(self.sp_dim)
        self.perm_con = np.arange(self.sp_dim)
        np.random.shuffle(self.perm_ant)
        np.random.shuffle(self.perm_con)

        self.perm_ant_inv = np.argsort(self.perm_ant)
        self.perm_con_inv = np.argsort(self.perm_con)

    def initialize_mtr_vocab(self, mtr_dim, mtr_sps):
        self.mtr_dim = mtr_dim

        self.mtr = Vocabulary(self.mtr_dim)
        for i, sp_str in enumerate(self.num_sp_strs):
            self.mtr.add(sp_str, mtr_sps[i, :])

        self.mtr_unk = Vocabulary(self.mtr_dim)
        self.mtr_unk.add(self.mtr_sp_strs[0], mtr_sps[-1, :])

        self.mtr_disp = self.mtr.create_subset(self.num_sp_strs)
        self.mtr_disp.readonly = False
        # Disable read-only flag for display vocab so that things can be added
        self.mtr_disp.add(self.mtr_sp_strs[0],
                          self.mtr_unk[self.mtr_sp_strs[0]].v)

    def initialize_vis_vocab(self, vis_dim, vis_sps):
        self.vis_dim = vis_dim

        self.vis = Vocabulary(self.vis_dim)
        for i, sp_str in enumerate(self.vis_sp_strs):
            self.vis.add(sp_str, vis_sps[i, :])

    def parse_instr_sps_list(self, instr_sps_list):
        instr_sps_list_sp = self.main.parse('0')

        if len(instr_sps_list) > self.max_enum_list_pos:
            raise ValueError('Vocabulator: Too many sequential ' +
                             'instructions. Max: %d' %
                             self.max_enum_list_pos + ', Got: ' +
                             len(instr_sps_list))

        for i, instr_sps in enumerate(instr_sps_list):
            instr_sps_list_sp += (self.main.parse('POS%i' % (i + 1)) *
                                  self.parse_instr_sps(*instr_sps))
        return instr_sps_list_sp

    def parse_instr_sps(self, ant_sp='0', cons_sp='0'):
        # Note: The ant and con permutations are used here to separate the
        #       possible POS tags in the ant/con from the instruction POS tag.
        #       This permutation is not necessary if the instruction POS tags
        #       differ from the number-representation POS tags.
        return SemanticPointer(self.main.parse(ant_sp).v[self.perm_ant] +
                               self.main.parse(cons_sp).v[self.perm_con])

vocab = SpaunVocabulary()
