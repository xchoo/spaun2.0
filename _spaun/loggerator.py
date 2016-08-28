import os
from datetime import datetime


class SpaunLogger(object):
    def __init__(self):
        self.data_dir = ''
        self.log_filename = ''
        self.data_obj = None

    def initialize(self, data_dir='', log_filename='log.txt'):
        self.data_dir = data_dir
        self.log_filename = log_filename

        self.data_filename = \
            os.path.join(self.data_dir, self.log_filename)
        self.data_obj = open(self.data_filename, 'a')

        self.write_header()

    def write_header(self):
        self.data_obj.write('# Spaun Simulation Properties:\n')
        self.data_obj.write('# - Run datetime: %s\n' % datetime.now())
        self.data_obj.write('#\n')

    def write(self, str):
        if self.data_obj is not None:
            orig_closed_state = self.data_obj.closed
            if orig_closed_state:
                self.data_obj = open(self.data_filename, 'a')

            self.data_obj.write(str)

            if orig_closed_state:
                self.data_obj.close()

    def flush(self):
        self.data_obj.flush()

    def close(self):
        self.data_obj.close()

logger = SpaunLogger()
