import numpy as np
from collections import OrderedDict

from .loggerator import logger


class SpaunExperiment(object):
    def __init__(self):
        self.num_map = {'0': 'ZER', '1': 'ONE', '2': 'TWO', '3': 'THR',
                        '4': 'FOR', '5': 'FIV', '6': 'SIX', '7': 'SEV',
                        '8': 'EIG', '9': 'NIN'}
        self.num_rev_map = {}
        for key in self.num_map.keys():
            self.num_rev_map[self.num_map[key]] = key

        self.sym_map = {'[': 'OPEN', ']': 'CLOSE', '?': 'QM'}
        self.sym_rev_map = {}
        for key in self.sym_map.keys():
            self.sym_rev_map[self.sym_map[key]] = key

        # Use alphabetical numerical representation for output logging so as
        # to avoid confusion between input stimulus and spaun outputs
        self.num_out_list = ['z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                             '-']

        self.null_output = "="

        self.raw_seq_str = ''
        self.raw_instr_str = ''

        self.raw_seq_list = []
        self.stim_seq_list = []
        self.task_phase_seq_list = []
        self.instr_sps_list = []
        self.instr_dict = []

        self.learn_min_num_actions = 2
        self._num_learn_actions = self.learn_min_num_actions

        self.present_blanks = False
        self.present_interval = 0.2

        self.prev_t_ind = -1

    @property
    def num_learn_actions(self):
        return max(self._num_learn_actions, self.learn_min_num_actions)

    def write_header(self):
        logger.write('# Spaun Experiment Options:\n')
        logger.write('# -------------------------\n')
        for param_name in sorted(self.__dict__.keys()):
            param_value = getattr(self, param_name)
            if not callable(param_value):
                if (isinstance(param_value, list) and len(param_value) > 20) \
                   or (isinstance(param_value, np.ndarray) and
                       param_value.shape[0] > 20):
                    param_value = str(param_value[:20])[:-1] + '...'
                if param_name == 'instr_dict':
                    logger.write('# - %s =: \n' % param_name)
                    for key in param_value.keys():
                        logger.write('#     > %s: %s -> %s\n' %
                                     (key, param_value[key][0],
                                      param_value[key][1]))
                else:
                    logger.write('# - %s = %s\n' % (param_name, param_value))
        logger.write('# ------\n')
        logger.write('# Simulation time: %0.2fs\n' % self.get_est_simtime())
        logger.write('#\n')

    def parse_mult_seq(self, seq_str):
        mult_open_ind = seq_str.find('{')
        mult_close_ind = seq_str.find('}')
        mult_value_ind = seq_str.find(':')

        if mult_open_ind >= 0 and mult_close_ind >= 0 and mult_value_ind >= 0:
            return self.parse_mult_seq(
                seq_str[:mult_open_ind] +
                self.parse_mult_seq(seq_str[mult_open_ind + 1:]))
        elif (mult_close_ind >= 0 and mult_value_ind >= 0 and
              mult_value_ind < mult_close_ind):
            return (seq_str[:mult_value_ind] *
                    int(seq_str[mult_value_ind + 1:mult_close_ind]) +
                    seq_str[mult_close_ind + 1:])
        elif mult_open_ind == mult_close_ind == mult_value_ind == -1:
            return seq_str
        else:
            raise ValueError('Invalid multiplicative indicator format.')

    def parse_custom_tasks(self, seq_str):
        task_open_ind = seq_str.find('(')
        task_close_ind = -1
        rslt_str = ""

        learn_task_options = []
        num_learn_actions = 0

        while task_open_ind >= 0:
            rslt_str += seq_str[task_close_ind + 1: task_open_ind]

            task_opts_ind = seq_str.find(';', task_open_ind)
            task_close_ind = seq_str.find(')', task_open_ind)

            if (task_opts_ind < 0 or task_close_ind < 0) or \
               (task_close_ind < task_opts_ind):
                raise ValueError('Malformed custom task string.')

            task_str = seq_str[task_open_ind + 1:task_opts_ind]
            task_opts_str = seq_str[task_opts_ind + 1:task_close_ind]

            if task_str == "COUNT":
                # Format: (COUNT; NUMCOUNT)
                count_val = int(task_opts_str)
                start_val = int(np.random.random() *
                                (len(self.num_map) - count_val))
                new_task_str = ('A4[%d][%d]?' % (start_val, count_val) +
                                'X' * (count_val + 1))
            elif task_str == "LEARN":
                # Format: (LEARN; PROB1A+PROB1B+ ... +PROB1N, NUMTRIALS1;
                #                 PROB2A+PROB2B+ ... +PROB2N, NUMTRAILS2; ...)
                learn_opts = task_opts_str.split(';')
                num_trials = 0

                for opt in learn_opts:
                    opt_list = opt.split(',')
                    prob_list = opt_list[0].split('+')

                    learn_opts = []

                    # Append number of trials to learn options list
                    learn_opts.append(int(opt_list[1]))
                    num_trials += learn_opts[0]

                    # Get probability list
                    prob_list = np.array(map(float, prob_list))

                    # Get number of learning options
                    num_learn_actions = max(num_learn_actions, len(prob_list))

                    # If probabilities are > 1, assume on scale of 0 - 100
                    if np.where(prob_list > 1)[0].shape[0] > 0:
                        prob_list /= 100

                    # Append probability list to learn options list
                    learn_opts.append(prob_list)

                    # Append learn options list to config master list
                    learn_task_options.append(learn_opts)

                # Generate task string
                new_task_str = 'A2' + '?X.' * num_trials

            elif task_str == "QA":
                # Format: (QA; P or K; LIST_LEN; PROBE_IND (optional))
                qa_opts = task_opts_str.split(';')
                qa_type = str(qa_opts[0]).upper()
                qa_len = int(qa_opts[1])

                if len(qa_opts) > 2:
                    qa_ind = int(qa_opts[2])
                else:
                    qa_ind = None

                # Generate number list
                num_list = np.array(self.num_map.keys())
                np.random.shuffle(num_list)
                num_list = list(map(str, num_list))

                # Generate position / kind options
                if qa_ind is None:
                    len_list = np.arange(qa_len)
                    np.random.shuffle(len_list)
                else:
                    len_list = [qa_ind]

                pk_opt = 0
                if qa_type == 'P':
                    pk_opt = str(len_list[0] + 1)
                elif qa_type == 'K':
                    pk_opt = num_list[len_list[0]]
                else:
                    raise ValueError('Option "%s" unsupported for QA task.' %
                                     qa_type)

                # Generate task string
                new_task_str = 'A5[%s]%s[%s]?XX' % (''.join(num_list[:qa_len]),
                                                    qa_type, pk_opt)

            elif task_str == "RVC":
                # Format: (RVC; Q_STR; A_STR; NUM_PAIRS)
                # Where Q_STR and A_STR is in the format:
                # - lower case letters for constants
                # - upper case 'X' for variable
                # e.g. (RVC; aaXb, Xb)
                rvc_opts = task_opts_str.split(';')
                rvc_q_str = rvc_opts[0]
                rvc_a_str = rvc_opts[1]
                rvc_len = int(rvc_opts[2])

                # Random digit list
                num_list = np.array(self.num_map.keys())
                np.random.shuffle(num_list)

                # RVC string format mapping
                rvc_const_map = dict()
                rvc_var_map = dict()

                char_cnt = 0
                # Handle all lower case characters first
                for char in (rvc_q_str + rvc_a_str):
                    if char not in rvc_const_map and char.islower():
                        rvc_const_map[char] = num_list[char_cnt]
                        char_cnt += 1

                for char in (rvc_q_str + rvc_a_str):
                    if char not in rvc_var_map and char.isupper():
                        var_list = []
                        for i in range(rvc_len + 1):
                            var_list.append(num_list[char_cnt])
                            char_cnt = (char_cnt + 1) % len(num_list)
                            if char_cnt == 0:
                                np.random.shuffle(num_list)
                        rvc_var_map[char] = var_list

                # Generate task string
                new_task_str = 'A6'
                for i in range(rvc_len + 1):
                    new_task_str += '['
                    for char in rvc_q_str:
                        if char.islower():
                            new_task_str += str(rvc_const_map[char])
                        if char.isupper():
                            new_task_str += str(rvc_var_map[char][i])
                    new_task_str += ']'
                    if i < rvc_len:
                        new_task_str += '['
                        for char in rvc_a_str:
                            if char.islower():
                                new_task_str += str(rvc_const_map[char])
                            if char.isupper():
                                new_task_str += str(rvc_var_map[char][i])
                        new_task_str += ']'
                new_task_str += '?' + 'X' * (len(rvc_a_str) + 1)
            else:
                raise ValueError('Custom task string "%s" ' % task_str +
                                 'not supported.')

            rslt_str += new_task_str
            task_open_ind = seq_str.find('(', task_open_ind + 1)

        return (rslt_str + seq_str[task_close_ind + 1:], learn_task_options,
                num_learn_actions)

    def parse_instruction_str(self, instr_str):
        instr_dict = OrderedDict()

        if len(instr_str) <= 0:
            return instr_dict

        instrs = instr_str.split(';')
        for instr in instrs:
            instr_data = instr.split(':')
            instr_key = instr_data[0]
            instr_sps = instr_data[1].split(',')

            if len(instr_sps) != 2:
                raise ValueError('Spaun Experimenter: Malformed instruction' +
                                 ' options.')

            instr_dict[instr_key] = instr_sps
        return instr_dict

    def insert_mtr_wait_sym(self, raw_seq_list, num_mtr_responses,
                            present_interval, mtr_est_digit_response_time):
        # Add 0.5 second motor response minimum
        num_mtr_responses += 0.5
        est_mtr_response_time = (num_mtr_responses *
                                 mtr_est_digit_response_time)
        extra_spaces = int(est_mtr_response_time / present_interval)

        raw_seq_list.extend([None] * extra_spaces)

    def add_present_blanks(self, seq_list, instr_list):
        seq_list_new = []
        instr_list_new = []

        for i, c in enumerate(seq_list):
            seq_list_new.append(c)
            instr_list_new.append(instr_list[i])
            if c != '.' and c is not None:
                seq_list_new.append(None)
                instr_list_new.append(instr_list[i])

        return seq_list_new, instr_list_new

    def is_valid_class_char(self, c):
        return (c.isalnum()) or (c in ['_'])

    def parse_raw_seq(self, raw_seq_str, get_image_ind, get_image_label,
                      present_blanks, mtr_est_digit_response_time,
                      instruction_str, rng):
        (raw_seq, learn_task_options, num_learn_actions) = \
            self.parse_custom_tasks(self.parse_mult_seq(raw_seq_str))

        instr_dict = self.parse_instruction_str(instruction_str)

        hw_class = False  # Flag to indicate to use a hand written number
        fixed_num = False
        is_instr = False

        raw_seq_list = []
        stim_seq_list = []
        instr_seq_list = []

        prev_c = ''
        fixed_c = ''
        instr_c = ''
        class_c = ''
        value_maps = {}

        num_n = 0
        num_r = 0

        num_mtr_responses = 0.0

        for c in raw_seq:
            # Process motor response before descriptor tags
            if num_mtr_responses > 0 and c in ['%', '#', '<']:
                self.insert_mtr_wait_sym(raw_seq_list, num_mtr_responses,
                                         self.present_interval,
                                         mtr_est_digit_response_time)
                num_mtr_responses = 0

            # Process hardwired class descriptor tag (# ...)
            if hw_class:
                if (not self.is_valid_class_char(c) and len(class_c) <= 0):
                    raise ValueError('Malformed class number string.')
                elif self.is_valid_class_char(c):
                    class_c += c
                    continue
                else:
                    raw_seq_list.append('#' + class_c)

                    # Since any non-alphanum character terminates class
                    # descriptor, need to clear class_c here as well.
                    # Also need to handle special case of '#' terminating
                    # class decriptor
                    class_c = ''
                    if c != '#':
                        hw_class = False
                    if c not in ['[', ']', '.', '<', '>', ' %']:
                        continue
            elif c == '#':
                hw_class = True
                class_c = ''
                continue

            # Process fixed image index descriptor tags (< ... >)
            if fixed_num:
                if (not c.isdigit() and c not in ['>']) or \
                   (c == '>' and len(fixed_c) <= 0):
                    raise ValueError('Malformed fixed index number string.')
                elif c.isdigit():
                    fixed_c += c
                else:
                    raw_seq_list.append('<' + fixed_c)
                    fixed_num = False
                continue
            elif c == '<':
                fixed_num = True
                fixed_c = ''
                continue

            # Process instruction string tags (% ... %)
            if c == '%':
                if not is_instr:
                    instr_c = ''
                else:
                    raw_seq_list.append('%' + instr_c)
                is_instr = not is_instr
                continue
            elif is_instr:
                instr_c += c
                continue

            # Process non-tagged characters
            if not (hw_class or fixed_num or is_instr):
                if c == 'N':
                    num_n += 1
                    continue
                else:
                    cs = np.random.choice(list(self.num_map.keys()), num_n,
                                          replace=False)
                    for n in cs:
                        raw_seq_list.append(n)
                    num_n = 0

                if c == 'R':
                    num_r += 1
                    continue
                else:
                    cs = np.random.choice(list(self.num_map.keys()), num_r,
                                          replace=True)
                    for r in cs:
                        raw_seq_list.append(r)
                    num_r = 0

                if c == 'A':    # Clear the value maps for each task
                    value_maps = {}
                    num_n = 0
                    num_r = 0

                if c.islower():
                    if c not in value_maps:
                        value_maps[c] = np.random.choice(
                            list(self.num_map.keys()), 1, replace=True)[0]
                    c = value_maps[c]

                if c == 'X':
                    num_mtr_responses += 1
                    continue
                elif num_mtr_responses > 0:
                    self.insert_mtr_wait_sym(raw_seq_list, num_mtr_responses,
                                             self.present_interval,
                                             mtr_est_digit_response_time)
                    num_mtr_responses = 0

                # 'R' Option for memory task
                if c == 'B':
                    c = 'R'

                raw_seq_list.append(c)

        # Insert trailing motor response wait symbols
        self.insert_mtr_wait_sym(raw_seq_list, num_mtr_responses,
                                 self.present_interval,
                                 mtr_est_digit_response_time)

        # Default instruction SP is blank
        instr_c = ''

        # Process raw sequence list to get actual SP's
        for c in raw_seq_list:
            if c in self.sym_map:
                c = self.sym_map[c]
            if c in self.num_map:
                c = self.num_map[c]

            if (c is not None) and (c[0] == '#'):
                class_c = c[1:]
                img_ind = get_image_ind(class_c, rng)
                c = (img_ind, class_c)
            elif (c is not None) and (c[0] == '<'):
                fixed_c = c[1:]
                c = (int(fixed_c), str(get_image_label(int(fixed_c))))
            elif (c is not None) and (c[0] == '%'):
                instr_c = c[1:]
                continue

            # If previous character is identical to current character, insert a
            # space between them.
            if (c is not None and prev_c == c and not present_blanks) or \
               c == '.':
                stim_seq_list.append(None)
                instr_seq_list.append(instr_c)
                if c == '.':
                    continue

            stim_seq_list.append(c)
            instr_seq_list.append(instr_c)

            # Keep track of previous character (to insert spaces between)
            # duplicate characters
            prev_c = c

        # Insert blanks if present_blanks option is set
        if present_blanks:
            stim_seq_list, instr_seq_list = \
                self.add_present_blanks(stim_seq_list, instr_seq_list)

        # Generate task phase sequence list
        task_phase_seq_list = []

        task = ''
        prev_task = ''
        learn_trial_count = 0
        learn_set_count = 0
        learn_num_trials = 0
        for s in stim_seq_list:
            prev_task = task
            if s == 'A':
                task = 'X'
                learn_trial_count = 0
            elif task == 'X':
                if s == 'ZER':
                    task = 'W'
                elif s == 'ONE':
                    task = 'R'
                elif s == 'TWO':
                    task = 'L'
                elif s == 'THR':
                    task = 'M'
                elif s == 'FOR':
                    task = 'C'
                elif s == 'FIV':
                    task = 'A'
                elif s == 'SIX':
                    task = 'V'
                elif s == 'SEV':
                    task = 'F'
                else:
                    task = 'UNK'

            if task == 'L':
                # Special handling stuff for learning task
                if s == 'QM':
                    # Keep track of how many learning trials (within the
                    # current learning task) that has been processed.
                    learn_trial_count += 1
                if prev_task != task or learn_trial_count > learn_num_trials:
                    # Keep track of how many learning sets that has been
                    # processed
                    learn_set_count += 1
                    learn_trial_count = int(prev_task == task)
                    # If prev_task != 'L', need to ignore the A2 in the
                    # learning trial count
                    learn_num_trials = \
                        learn_task_options[learn_set_count - 1][0]

                learn_rewards = learn_task_options[learn_set_count - 1][1]
                task_phase_seq_list.append([task, learn_rewards])
            else:
                task_phase_seq_list.append(task)

        # Parse Instruction character key list into instruction sem pointer
        # list
        instr_sp_list = []
        instr_change_inds = {}
        prev_instr_sp = ''
        for i, key in enumerate(instr_seq_list):
            if key == '':
                instr_sp_list.append(None)
            else:
                instr_sp_sublist = []
                for sp_str in key.split('+'):
                    if sp_str in instr_dict.keys():
                        instr_sp_sublist.append(instr_dict[sp_str])
                    else:
                        raise ValueError('Spaun Experimenter: Instruction ' +
                                         'key "%s"' % sp_str + ' not found ' +
                                         'in provided instruction options.')
                instr_sp_list.append(instr_sp_sublist)

            # Record down when the instruction sp changes: for logging purposes
            if key != prev_instr_sp:
                instr_change_inds[i] = key
                prev_instr_sp = key

        return (raw_seq_list, stim_seq_list, task_phase_seq_list,
                instr_sp_list, instr_dict, instr_change_inds,
                num_learn_actions)

    def get_est_simtime(self):
        return (len(self.stim_seq_list) * self.present_interval)

    def get_t_ind_float(self, t):
        return (t / self.present_interval / (2 ** self.present_blanks))

    def get_t_ind(self, t):
        return int(self.get_t_ind_float(t))

    def in_learning_phase(self, t):
        t_ind = min(self.get_t_ind(t), len(self.task_phase_seq_list) - 1)
        task = self.task_phase_seq_list[t_ind]
        return (len(task) > 1 and task[0] == 'L')

    def get_stimulus(self, t):
        t_ind = self.get_t_ind(t)
        t_ind_float = self.get_t_ind_float(t)

        if t <= 0:
            return None

        if t_ind != self.prev_t_ind:
            # Write the stimulus to file
            if t_ind < len(self.stim_seq_list):
                # Log instruction sp changes
                if t_ind in self.instr_change_inds.keys():
                    logger.write('\n>%s' % self.instr_change_inds[t_ind])

                stim_char = self.stim_seq_list[t_ind]
                if (stim_char == '.'):
                    # logger.write('_')
                    logger.write('')  # Ignore the . blank character
                elif stim_char in ['A', 'M', 'V']:  # and self.prev_t_ind >= 0:
                    logger.write('\n' + stim_char)
                elif isinstance(stim_char, int):
                    logger.write('<%s>' % stim_char)
                elif stim_char in self.num_rev_map:
                    logger.write('%s' % self.num_rev_map[stim_char])
                elif stim_char in self.sym_rev_map:
                    logger.write('%s' % self.sym_rev_map[stim_char])
                elif stim_char is not None:
                    logger.write('%s' % str(stim_char))

            # Done all the stuff needed for new t_ind. Store new t_ind
            self.prev_t_ind = t_ind

        if (self.present_blanks and t_ind != int(round(t_ind_float))) or \
           t_ind >= len(self.stim_seq_list) or \
           self.stim_seq_list[t_ind] == '.':
            return None
        else:
            return self.stim_seq_list[t_ind]

    def get_instruction_sps(self, t):
        t_ind = self.get_t_ind(t)
        if t_ind < len(self.instr_sps_list):
            return self.instr_sps_list[t_ind]
        else:
            return None

    def update_output(self, t, out_ind):
        # Figure out what the motor output is and write it to file
        if out_ind >= 0 and out_ind < len(self.num_out_list):
            out_str = self.num_out_list[out_ind]
        else:
            out_str = self.null_output
        logger.write(out_str)
        logger.flush()

        if self.in_learning_phase(t):
            # Denote learning phase reward
            logger.write('|')
            logger.flush()

            # In learning phase. Evaluate output and choose reward
            if out_ind >= 0 and out_ind < (len(self.num_out_list) - 1):
                t_ind = min(self.get_t_ind(t),
                            len(self.task_phase_seq_list) - 1)
                rewards = self.task_phase_seq_list[t_ind][1]

                if out_ind >= 0 and out_ind < self.num_learn_actions:
                    reward_chance = rewards[out_ind]
                else:
                    reward_chance = 0

                rewarded = str(int(np.random.random() < reward_chance))
                self.stim_seq_list[self.get_t_ind(t) + 1] = \
                    self.num_map[rewarded]
            elif (self.get_t_ind(t) + 1) < len(self.stim_seq_list):
                self.stim_seq_list[self.get_t_ind(t) + 1] = self.num_map['0']
        else:
            pass

    def initialize(self, raw_seq_str, get_image_ind, get_image_label,
                   mtr_est_digit_response_time, instruction_str, rng):
        self.raw_seq_str = raw_seq_str.replace(' ', '')
        self.raw_instr_str = instruction_str.replace(' ', '')

        (self.raw_seq_list, self.stim_seq_list, self.task_phase_seq_list,
         self.instr_sps_list, self.instr_dict, self.instr_change_inds,
         self._num_learn_actions) = \
            self.parse_raw_seq(self.raw_seq_str, get_image_ind,
                               get_image_label, self.present_blanks,
                               mtr_est_digit_response_time,
                               self.raw_instr_str, rng)

    def reset(self):
        self.prev_t_ind = -1

experiment = SpaunExperiment()
