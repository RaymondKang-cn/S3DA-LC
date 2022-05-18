import torch.nn as nn
from customlayers import Classifiers
from customlayers import BackBoneLayer
from customlayers import ForwardLayer
import config as config


class MultiSourceNet(nn.Module):

    def __init__(self):

        super(MultiSourceNet, self).__init__()

        self.model = {}
        for module in config.settings['net_list']:
            if module == 'B':
                self.model[module] = BackBoneLayer(config.settings['bb'], config.settings['bb_output'])
            elif module == 'F':
                self.model[module] = ForwardLayer(config.settings['bb_output'], config.settings['bb_output'] // 2,
                                                  config.settings['F_dims'])
            elif module == 'C':
                self.model[module] = Classifiers(config.settings['F_dims'], len(config.settings['src_datasets']),
                                                 config.settings['num_C'][config.settings['src_datasets'][0]])

        for name, comp in self.model.items():
            self.add_module('_'.join([name]), comp)

    def forward(self, x):
        raise NotImplementedError('Implemented a custom forward in train loop')


if __name__ == '__main__':
    raise NotImplementedError('Please check README.md for execution details')
