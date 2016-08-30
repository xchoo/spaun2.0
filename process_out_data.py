import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from _spaun.utils import conf_interval


parser = argparse.ArgumentParser(description='Script for analyzing spaun2.0' +
                                 'results.')
parser.add_argument('--data_dir', type=str, default='data',
                    help='Probe directory.')
parser.add_argument('-p', type=str, default='probe_data',
                    help='Probe data filename prefix. E.g. probe_data')
parser.add_argument('-n', type=str, default='LIF_512',
                    help='Probe data neuron type str. E.g. LIF_512')
parser.add_argument('-t', type=str, default=None,
                    help='Probe data tag str.')
parser.add_argument('-s', type=str, default='',
                    help='Probe data stimulus str. E.g. A0[0]@XXX')
parser.add_argument('--output_file', type=str, default=None,
                    help='Ouput data file name.')
parser.add_argument('-a', action='store_true',
                    help='Supply to append data to output file. Default is ' +
                    'to overwrite the file.')
parser.add_argument('-r', action='store_true',
                    help='Supply to read data from output file. No ' +
                    'additional log file processing is done.')

args = parser.parse_args()


response_strs = ['z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', '-', '=']
num_list_strs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '-']


def mass_str_replace(input_str, search_list, replace_list):
    # Note: replace_list must be the same len as search_list, or a single
    #       string.
    if not isinstance(replace_list, str) and (len(search_list) !=
                                              len(replace_list)):
        raise RuntimeError('Mismatching replace and search list terms')

    # Make a single string replace_list into a list
    if isinstance(replace_list, str):
        replace_list = [replace_list] * len(search_list)

    # Do the string replacement
    for i, item in enumerate(search_list):
        input_str = input_str.replace(item, replace_list[i])

    return input_str


def remove_MNIST_strs(task_info_str):
    str_split = task_info_str.split('(')

    for i, sub_str in enumerate(str_split):
        if ')' in sub_str:
            sub_str_split = sub_str.split(',', 1)
            str_split[i] = sub_str_split[1][:-2].strip()

    return ''.join(str_split)


# Process probe data file entry
def process_line(task_str, task_data_str):
    # Ignore any responses that make it into the task string
    task_str = mass_str_replace(task_str, response_strs, '')

    # Process task_data_str into component bits
    # For all tasks except learning task, extract spaun's answer
    if task_str in ['A0', 'A1', 'A3', 'A4', 'A5', 'A6', 'A7']:
        # Split the task data string into before and after the question mark
        task_data_split = task_data_str.split('?', 1)

        # The task information is before the question mark
        task_info = task_data_split[0].replace("'", '')
        # Filter out the MNIST digits
        task_info = remove_MNIST_strs(task_info)

        # Record special characters
        has_F = 'F' in task_info
        has_R = 'R' in task_info
        has_P = 'P' in task_info
        has_K = 'K' in task_info

        # Split up the different components of the task info
        task_info_split = task_info.split(']')

        if task_info_split[-1] == '':
            task_info_split = task_info_split[:-1]

        # Remove [ ]'s and special characters from each part of task_info_split
        for i in range(len(task_info_split)):
            task_info_split[i] = \
                mass_str_replace(task_info_split[i],
                                 ['[', ']', 'F', 'R', 'P', 'K'], '')

        # Spaun's answer is after the question mark
        task_answer_spaun = \
            np.array(list(mass_str_replace(task_data_split[1],
                                           response_strs, num_list_strs)))

        if len(task_answer_spaun) == 0:
            return (None, None)

    # ------ Reference answer generation ------
    if task_str in ['A0', 'A1', 'A3']:
        # For copy-draw, classification, memory task
        task_info = np.array(list(task_info_split[0]))
        if has_R:
            task_answer_ref = task_info[-1::-1]
        else:
            task_answer_ref = task_info
    elif task_str == 'A4':
        # For counting tasks
        start_num = int(task_info_split[0])
        count_num = int(task_info_split[1])
        ans_num = start_num + count_num

        # Ignore invalid task options
        if ans_num > 9:
            task_str = 'INVALID'
            warn('A4: Computed answer > 9')

        task_answer_ref = np.array([str(ans_num)])
    elif task_str == 'A5':
        # QA task
        num_list = map(int, list(task_info_split[0]))
        probe_num = int(task_info_split[1])

        if has_P:
            task_answer_ref = np.array([str(num_list[probe_num - 1])])
        elif has_K:
            task_answer_ref = np.array([str(num_list.index(probe_num) + 1)])
        else:
            task_str = 'INVALID'
            warn('A5: No valid P/K for QA task')
    elif task_str == 'A6':
        from sets import Set
        # RVC task
        if len(task_info_split) % 2:
            match_list = None
            for i in range(len(task_info_split) / 2):
                list1 = np.array(list(task_info_split[i * 2]))
                list2 = np.array(list(task_info_split[i * 2 + 1]))
                if match_list is None:
                    match_list = [Set(np.where(list1 == item)[0])
                                  for item in list2]
                else:
                    # TODO: Check for inconsistencies across pairs
                    if len(list2) != len(match_list):
                        warn('A6: Inconsistent RVC ref answer lengths.')
                        task_str = 'INVALID'
                    else:
                        match_list = [match_list[j] &
                                      Set(np.where(list1 == list2[j])[0])
                                      for j in range(len(match_list))]
            list1 = np.array(list(task_info_split[-1]))
            task_answer_ref = np.array([list1[list(set_list)[0]]
                                        for set_list in match_list])
        else:
            task_str = 'INVALID'
            warn('A6: Invalid RVC task. No question list given.')
    elif task_str == 'A7':
        # Raven's induction task
        # Induction task comes in two forms: changing list len, and changing
        #                                    number relations
        col_count = 1
        induction_diff = None
        induction_len_change = None

        for i in range(1, len(task_info_split)):
            if col_count % 3 == 0:
                col_count += 1
                continue
            list1 = map(int, np.array(list(task_info_split[i - 1])))
            list2 = map(int, np.array(list(task_info_split[i])))

            # Handle the following cases:
            # 1. Unchanging list lengths of len 1
            if len(list1) == len(list2) == 1:
                diff = list2[0] - list1[0]
                if induction_diff is None:
                    induction_diff = diff
                if induction_diff != diff:
                    warn('A7: Inconsistent change between induction items')
                    task_str = 'INVALID'
            # 2. Changing list lengths, but containing identical items
            elif list1[0] == list2[0]:
                len_change = len(list2) - len(list1)
                if induction_len_change is None:
                    induction_len_change = len_change
                if induction_len_change != len_change:
                    warn('A7: Inconsistent change between list lenghts')
                    task_str = 'INVALID'
            else:
                warn('A7: Unhandled induction task type')
                task_str = 'INVALID'

            # Handle transition to next row
            col_count += 1

        def spaun_response_to_int(c):
            return int(c) if c.isdigit() else -1

        list1 = map(spaun_response_to_int, list(task_info_split[-1]))
        if induction_diff is not None and induction_len_change is None:
            task_answer_ref = np.array(map(str, [list1[0] + induction_diff]))
        elif induction_len_change is not None and induction_diff is None:
            task_answer_ref = np.array(map(str, [list1[0]] * (len(list1) + 1)))
        else:
            warn('A7: Multiple induction types encountered?')
            task_str = 'INVALID'

    # Format the task answer list (make the same length as the reference
    # answer list). Applies to all but learning task
    if task_str == 'INVALID':
        return task_str, np.array([0])

    if task_str in ['A0', 'A1', 'A3', 'A4', 'A5', 'A6', 'A7']:
        task_answer = np.chararray(task_answer_ref.shape)
        task_answer[:] = ''
        task_answer_len = min(len(task_answer_ref), len(task_answer_spaun))
        task_answer[:task_answer_len] = task_answer_spaun[:task_answer_len]

        # DEBUG
        # print task_data_str, task_answer, task_answer_ref
    else:
        print task_data_str

    if task_str in ['A0', 'A1', 'A3']:
        # For memory, recognition, copy drawing tasks, check recall accuracy
        # per item
        return ('_'.join([task_str, str(len(task_answer_ref))]),
                map(int, task_answer == task_answer_ref))

    if task_str in ['A4', 'A5', 'A6', 'A7']:
        # For other non-learning tasks, check accuracy as wholesale correct /
        # incorrect
        return ('_'.join([task_str, str(len(task_answer_ref))]),
                [int(np.all(task_answer == task_answer_ref))])


# Process probe data file
probe_dir = args.data_dir
str_prefix = '+'.join([args.p, args.n])
if len(args.s) > 0:
    str_prefix = '+'.join([str_prefix, args.s])
if args.t is not None:
    str_suffix = '(' + args.t + ')_log.txt'
else:
    str_suffix = '_log.txt'

processed_results = {}

for filename in os.listdir(probe_dir):
    if filename[-len(str_suffix):] == str_suffix and \
       filename[:len(str_prefix)] == str_prefix and not args.r:
        print "PROCESSING: " + os.path.join(probe_dir, filename)
        probe_file = open(os.path.join(probe_dir, filename), 'r')
        for line in probe_file.readlines():
            if line[0] != '#' and line.strip() != '':
                task_info_split = line.split('[', 1)
                task_str = task_info_split[0].strip()
                task_data = task_info_split[1].strip()

                task_str, task_result = process_line(task_str, task_data)

                if task_str is not None:
                    if task_str not in processed_results:
                        processed_results[task_str] = [task_result]
                    else:
                        processed_results[task_str].append(task_result)

# Convert all data structures in processed results to np arrays
for task in processed_results:
    processed_results[task] = np.array(processed_results[task])

# Write collected data to file
if args.output_file is None:
    output_file = '+'.join(['results', str_prefix]) + '.npz'
else:
    output_file = args.output_file

output_filepath = os.path.join(probe_dir, output_file)
if args.a or args.r:
    old_result_data = np.load(output_filepath)
    old_results = dict(old_result_data)

    for key in processed_results:
        if key in old_results:
            old_data = old_results[key]
            combined_dim = processed_results[key].shape[1]
            combined_len = (processed_results[key].shape[0] +
                            old_data.shape[0])
            combined_results = np.zeros((combined_len, combined_dim))
            combined_results[:old_data.shape[0], :] = old_data
            combined_results[old_data.shape[0]:, :] = processed_results[key]
            old_results[key] = combined_results
        else:
            old_results[key] = processed_results[key]

    processed_results = old_results

if not args.r:
    # Write new data to file
    np.savez_compressed(output_filepath, **processed_results)

# DEBUG
# for key in processed_results:
#     print ">>>>> %s <<<<<" % key
#     for d in processed_results[key]:
#         print d

# Compute CI and plot data
ci_data_filepath = output_filepath[:-4] + '_ci.npz'
if not args.r:
    ci_data = {}
    for key in processed_results:
        results = processed_results[key]
        ci_data[key] = \
            np.array([list(conf_interval(results[:, i]))
                      for i in range(results.shape[1])])
        # Format: [0]: mean, [1]: low, [2]: high

    # Write CI data to file
    np.savez_compressed(ci_data_filepath, **ci_data)
else:
    ci_data = dict(np.load(ci_data_filepath))

# Print CI data
print ci_data

# Plot results
for task in ci_data:
    data = ci_data[task]
    xvals = np.arange(data.shape[0]) + 1
    means = data[:, 0]
    lows = data[:, 1]
    highs = data[:, 2]

    plt.figure(figsize=(18, 9))
    plt.errorbar(xvals, means, yerr=[means - lows, highs - means])
    plt.xlim(0.5, data.shape[0] + 0.5)
    plt.ylim(0, 1)

plt.show()
