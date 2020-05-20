# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class RuntimeStats:
    def __init__(self, forward):
        self.stats = {
            'compute_time': 0.0,
            'send_tensors': 0.0,
            'send_tensors_size': 0,
            'receive_tensors': 0.0,
            'receive_tensors_size': 0,
        }
        self.forward = forward
        self.units = []
        for i in self.stats:
            self.units.append('s')
            if i == 'receive_tensors_size' or i == 'send_tensors_size':
                self.units.append('b')

    def print_stats(self):
        if self.forward:
            s = "Forward Stats:"
        else:
            s = "Backward Stats:"
        for i, (k, v) in enumerate(self.stats.items()):
            s += "\t %s %.3f %s" % (k, v, self.units[i])
        print(s)

    def reset_stats(self):
        for i in self.stats.keys():
            self.stats[i] = 0.0
