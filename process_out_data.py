import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Script for analyzing spaun2.0' +
                                 'results.')
parser.add_argument('-d', type=str, default='.',
                    help='Probe directory.')
parser.add_argument('-p', type=str, default='probe_data',
                    help='Probe data filename prefix. E.g. probe_data')
parser.add_argument('-n', type=str, default='LIF_512',
                    help='Probe data neuron type str. E.g. LIF_512')
parser.add_argument('-s', type=str, default='',
                    help='Probe data stimulus str. E.g. A0[0]@XXX')

args = parser.parse_args()


# Process probe data file entry
def process_line(task_str, task_data, task_answer):
    if task_str in ['A0', 'A1', 'A3']:
        task_data = task_data.replace('[', '').replace(']', '')
        task_data = np.array(list(task_data))

        task_answer_source = np.array(list(task_answer))
        task_answer = np.chararray(task_data.shape)
        task_answer[:] = ''
        task_answer[:len(task_answer_source)] = \
            task_answer_source[:min(len(task_answer_source), len(task_data))]

        return ('_'.join([task_str, str(len(task_data))]),
                map(int, task_data == task_answer))
    elif task_str == 'A1':
        pass
    elif task_str == 'A2':
        pass
    elif task_str == 'A4':
        pass
    elif task_str == 'A5':
        pass
    elif task_str == 'A6':
        pass
    elif task_str == 'A7':
        pass
    else:
        return task_str, np.array([0])

# Process probe data file
probe_dir = args.d
str_prefix = '+'.join([args.p, args.n])
if len(args.s) > 0:
    str_prefix = '+'.join([str_prefix, args.s])
str_suffix = '_log.txt'

processed_results = {}

for filename in os.listdir(probe_dir):
    if filename[-len(str_suffix):] == str_suffix and \
       filename[:len(str_prefix)] == str_prefix:
        probe_file = open(os.path.join(probe_dir, filename), 'r')
        for line in probe_file.readlines():
            if line[0] != '#':
                qa_split = line.split('?')
                if len(qa_split) == 2:
                    task_answer = qa_split[1]
                    task_info = qa_split[0].split('[', 1)
                    if len(task_info) == 2:
                        task_str = task_info[0]
                        task_data = task_info[1]
                        task_str, task_result = \
                            process_line(task_str, task_data, task_answer)

                        if task_str not in processed_results:
                            processed_results[task_str] = [task_result]
                        else:
                            processed_results[task_str].append(task_result)

print processed_results
