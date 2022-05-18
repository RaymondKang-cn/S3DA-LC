import json
import time
from numpy.core.fromnumeric import ptp
import torch
import os
import torch.optim as optim
from tabulate import tabulate
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as config
from dataset import TemplateDataset, PseudoTargetDataset
from net import MultiSourceNet as ms_model
from sklearn.metrics import f1_score


class TrainerG(object):

    def __init__(self):

        self.settings = config.settings
        self.weight = torch.Tensor([1 for _ in range(len(config.settings['src_datasets']))])
        self.class_threshold = None

        self.network = ms_model().to(self.settings['device'])
        self.optimizer = None
        self.init_optimizers()

        # batch size
        self.batch_size = self.settings['batch_size']
        self.val_batch_size = self.settings['batch_size'] * self.settings['val_batch_size_factor']

        self.current_iteration = self.settings['start_iter']
        self.exp_name = self.settings['exp_name']
        self.phase = self.settings['mode']['train']

        self.src_data = {}
        self.src_features = {}
        self.pseudo_data = {}
        self.pseudo_features = {}

        # data loader dictionaries
        self.source_dl_iter_train_list = {}
        self.target_dl_iter_train_list = {}
        self.source_dl_iter_val_list = {}
        self.target_val_list = {}

        # data structure to hold pseudo target labels
        self.pseudo_target_dl_iter_train_list = {}
        self.pseudo_idx_list = {}
        self.pseudo_cls_label_list = {}
        self.confidence_threshold = torch.ones(config.settings['class_num'])

        self.max_acc = -10000

        self.get_all_train_src_dataloaders()

        self.start_time = time.strftime("%d_%H_%M", time.localtime())

        self.cal_metrics = {'net': config.args.net,
                            'tau': config.args.tau,
                            'use_source': config.args.use_source,
                            'fix': config.args.fix,
                            'alpha': config.args.alpha,
                            'use_weight': config.args.use_weight,
                            'max_micro_score': -1,
                            'max_macro_score': -1,
                            'micro_score': [],
                            'macro_score': [],
                            'micro_score_p': [],
                            'macro_score_p': [],
                            'pseudo_rate': [],
                            'update_iter': [],
                            'class_wise_p': [],
                            'class_wise_orl': [],
                            'class_wise_true': [],
                            'domain_weight': [],
                            'class_threshold': [],
                            'domain_wise_acc': [[] for _ in range(len(config.settings['src_datasets']))],
                            }
        self.itt_delete = []

    '''
    Utility function to implement logic while saving weights
    '''

    def check_and_save_weights(self, curr_cls_acc):
        self.max_acc = float(max(curr_cls_acc, self.max_acc))
        if self.max_acc == curr_cls_acc:
            self.save_weights()
            self.itt_delete.append(self.current_iteration)
            if len(self.itt_delete) > self.settings['checkpoints_count']:
                for k in self.itt_delete[:-self.settings['checkpoints_count']]:
                    os.remove(os.path.join('exp', self.exp_name, 'model_' + str(k) + '.pth'))
                    os.remove(os.path.join('exp', self.exp_name, 'opt_' + str(k) + '.pth'))
                self.itt_delete = self.itt_delete[-self.settings['checkpoints_count']:]

        if self.current_iteration == self.settings['enough_iter']:
            self.save_weights()

    '''
        Function to save model and optimizer state
    '''

    def save_weights(self):
        print('saving best weight at iteration number ={}'.format(
            self.current_iteration))

        model_state_dict = {}
        for name, comp in self.network.model.items():
            model_state_dict[name] = comp.cpu().state_dict()
        if not os.path.exists(os.path.join('exp', self.exp_name)):
            os.mkdir(os.path.join('exp', self.exp_name))
        if self.current_iteration == self.settings['enough_iter']:
            torch.save(model_state_dict, os.path.join('exp',
                                                      self.exp_name,
                                                      'model_enough_iter' + str(self.current_iteration) + '.pth'))
        torch.save(model_state_dict, os.path.join('exp',
                                                  self.exp_name, 'model_' + str(self.current_iteration) + '.pth'))

        optimizer_state = self.optimizer.state_dict()
        if self.current_iteration == self.settings['enough_iter']:
            torch.save(optimizer_state, os.path.join('exp',
                                                     self.exp_name,
                                                     'opt_enough_iter' + str(self.current_iteration) + '.pth'))
        torch.save(optimizer_state, os.path.join('exp', self.exp_name,
                                                 'opt_' + str(self.current_iteration) + '.pth'))
        self.network.to(self.settings['device'])

    '''
        Function to load model weights
    '''

    def load_model(self):

        model_path = os.path.join('exp', self.settings['model_dict']['exp_name'],
                                  'model_' + str(self.settings['model_dict']['iter']) + '.pth')
        model_state_dict = torch.load(model_path, map_location=self.settings['device'])
        for name in config.settings['net_list']:
            self.network.model[name].load_state_dict(model_state_dict[name])

    '''
    Function to load optimizer
    '''

    def load_optimizers(self):
        opt_path = os.path.join('exp', self.settings['opt_dict']['exp_name'],
                                'opt_' + str(self.settings['opt_dict']['iter']) + '.pth')
        optimizer_state = torch.load(opt_path, map_location=self.settings['device'])
        self.optimizer.load_state_dict(optimizer_state)
        self.current_iteration = self.settings['opt_dict']['iter']

    '''
    Initializing optimizers
    '''

    def init_optimizers(self):
        opt_param_list = []
        for name in self.settings['net_list']:
            if name == 'B':
                opt_param_list.append({'params': self.network.model[name].parameters(),
                                       'lr': self.settings['lr'] / 10.0,
                                       'weight_decay': 5e-4})
            else:
                opt_param_list.append({'params': self.network.model[name].parameters(),
                                       'lr': self.settings['lr'], 'weight_decay': 5e-4})
        self.optimizer = optim.Adam(params=opt_param_list)

    '''
    Initialize pseudo target labels
    '''

    def initialize_pseudo_target_indices(self):
        self.network.eval()

        print("\n----------- Calculating pseudo labels at iteration " + str(self.current_iteration) + ' -----------\n')
        print('Running dataset = {}, task = {}, net = {}'.format(
            config.dataset, config.task, config.args.net))
        print('Parameter tau = {}, fix = {}, alpha = {}'.format(
            config.args.tau, config.args.fix, config.args.alpha))
        self.cal_metrics['update_iter'].append(self.current_iteration)

        with torch.no_grad():
            for dom in self.settings['trgt_datasets']:

                target_dataset_train = TemplateDataset('_'.join([dom, 'train.npy']), aug=False)

                target_dl_iter_train_list = iter(
                    DataLoader(target_dataset_train, batch_size=self.val_batch_size, shuffle=False,
                               num_workers=config.args.worker,
                               pin_memory=True))

                all_indices = []
                all_logits = [[] for _ in range(len(self.settings['src_datasets']))]
                all_labels = []

                for data in tqdm(target_dl_iter_train_list):
                    index, images, labels, _ = data
                    images = images.to(self.settings['device']).float()
                    labels.long()
                    feats = self.network.model['F'](self.network.model['B'](images))
                    logits = self.network.model['C'](feats, 'eval')

                    for i in range(len(self.settings['src_datasets'])):
                        all_logits[i].append(logits[i])  # len(self.settings['src_datasets']) * target size * classes
                    all_indices.append(index)
                    all_labels.append(labels)

                domain_wise_preds = []
                domain_wise_ent = []
                for i in range(len(self.settings['src_datasets'])):
                    all_logits[i] = torch.cat(all_logits[i], dim=0).cpu()
                    domain_wise_preds.append(all_logits[i].softmax(dim=1))
                    ent = torch.sum(-domain_wise_preds[i] * torch.log(domain_wise_preds[i]), dim=1).mean()
                    domain_wise_ent.append(float(1 / ent))
                weight = torch.Tensor(domain_wise_ent) / torch.max(torch.Tensor(domain_wise_ent))
                if self.current_iteration == self.settings['enough_iter'] + 1:
                    self.weight = weight
                else:
                    if config.args.alpha > 0:
                        self.weight = config.args.alpha * self.weight + (
                                1 - config.args.alpha) * weight
                    else:
                        self.weight = self.cal_metrics['pseudo_rate'][-1] * self.weight + (
                                1 - self.cal_metrics['pseudo_rate'][-1]) * weight

                all_indices = torch.cat(all_indices, dim=0).cpu()
                all_labels = torch.cat(all_labels, dim=0).cpu()

                if config.args.use_weight == 0:
                    self.weight = torch.Tensor([1 for _ in range(len(config.settings['src_datasets']))])
                
                weight_logits = [self.weight[i] / len(self.settings['src_datasets']) * all_logits[i] for i in
                                 range(len(self.settings['src_datasets']))]
                all_logits = 0
                for i in range(len(self.settings['src_datasets'])):
                    all_logits += weight_logits[i]
                all_preds = all_logits.softmax(dim=1)
                pred_labels = torch.argmax(all_preds, dim=1)

                sample_num = torch.zeros(config.settings['class_num'])
                i = 0
                for d in self.settings['src_datasets']:
                    sample_num += self.weight[i] * torch.Tensor(config.settings['sample_num'][config.dataset][d])
                    i = i + 1
                sample_num = sample_num / torch.sum(sample_num)

                class_wise_num = []
                for i in range(self.settings['class_num']):
                    class_wise_num.append((pred_labels == i).sum() / len(all_preds))
                if config.args.use_source == 1:
                    class_threshold = torch.Tensor(class_wise_num) / sample_num
                else:
                    class_threshold = torch.Tensor(class_wise_num)
                class_threshold = config.args.tau * class_threshold / torch.max(class_threshold)

                if config.args.fix:
                    class_threshold = torch.ones(len(class_wise_num)) * config.args.tau

                if self.current_iteration == self.settings['enough_iter'] + 1:
                    self.class_threshold = class_threshold
                    print(self.class_threshold)
                else:
                    if config.args.alpha > 0:
                        self.class_threshold = config.args.alpha * self.class_threshold + (
                                1 - config.args.alpha) * class_threshold
                    else:
                        self.class_threshold = self.cal_metrics['pseudo_rate'][-1] * self.class_threshold + (
                                1 - self.cal_metrics['pseudo_rate'][-1]) * class_threshold

                above_threshold = all_preds > class_threshold

                for i in range(len(all_labels)):
                    if not above_threshold[i][pred_labels[i]]:
                        pred_labels[i] = -1
                indices = pred_labels > -1
                pseudo_labels = pred_labels[indices]
                indices = all_indices[indices]

            self.pseudo_cls_label_list[dom], self.pseudo_idx_list[dom] = pseudo_labels, indices

            self.for_record(pseudo_labels, all_labels[indices], dom)
            class_wise_orl = []
            class_wise_num = []
            class_wise_true = []
            for i in range(self.settings['class_num']):
                class_wise_orl.append(int((torch.argmax(all_preds, dim=1) == i).sum()))
                class_wise_num.append(int((pseudo_labels == i).sum()))
                class_wise_true.append(int((all_labels[indices] == i).sum()))
            sort_index = torch.argsort(torch.Tensor(class_wise_orl), descending=True)
            self.cal_metrics['class_wise_p'].append(class_wise_num)
            self.cal_metrics['class_wise_orl'].append(class_wise_orl)
            self.cal_metrics['class_wise_true'].append(class_wise_true)
            # print(torch.Tensor(class_wise_orl)[sort_index][0:10])
            # print(torch.Tensor(class_wise_num)[sort_index][0:10])
            # print(self.weight)

        self.network.train()
        self.save_log()
        print('setting model in mode \'train\'')

    '''
    Calculate cal_metric
    '''

    def for_record(self, pred_labels, true_labels, dom):
        micro_score = f1_score(true_labels, pred_labels, average='micro')
        macro_score = f1_score(true_labels, pred_labels, average='macro')
        pseudo_rate = float(len(true_labels) / sum(config.settings['sample_num'][config.dataset][dom]))
        met = [['Pseudo rate', pseudo_rate],
               ['Pseudo label macro score', macro_score],
               ['Pseudo label micro score', micro_score]]

        print(tabulate(met, headers=['metrics', 'value'], tablefmt="simple", showindex=True))
        self.cal_metrics['micro_score_p'].append(micro_score)
        self.cal_metrics['macro_score_p'].append(macro_score)
        self.cal_metrics['pseudo_rate'].append(pseudo_rate)
        self.cal_metrics['domain_weight'].append(self.weight.numpy().tolist())
        self.cal_metrics['class_threshold'].append(self.class_threshold.numpy().tolist())

    '''
    Save Log
    '''

    def save_log(self):
        self.cal_metrics['max_micro_score'] = max(self.cal_metrics['micro_score'])
        self.cal_metrics['max_macro_score'] = max(self.cal_metrics['macro_score'])
        filename = './results/' + str(self.settings['dataset_name']) + '/' + str(self.settings['data_key']) + '(' + str(
            self.start_time) + ').json'
        with open(filename, 'w') as f:
            json.dump(self.cal_metrics, f)

    '''
    Utility Functions to initialize source and target dataloaders
    '''

    def get_all_train_src_dataloaders(self):
        for dom in self.settings['src_datasets']:
            self.initialize_src_train_dataloader(dom)

    def initialize_src_train_dataloader(self, dom):
        source_dataset_train = TemplateDataset('_'.join([dom, 'train.npy']), aug=True)
        self.source_dl_iter_train_list[dom] = iter(DataLoader(source_dataset_train,
                                                              batch_size=self.batch_size,
                                                              shuffle=True,
                                                              num_workers=config.args.worker,
                                                              drop_last=True,
                                                              pin_memory=True))

    def get_all_val_target_dataloaders(self):
        for dom in self.settings['trgt_datasets']:
            self.initialize_target_val_dataloader(dom)

    def initialize_target_val_dataloader(self, dom):
        target_dataset_val = TemplateDataset('_'.join([dom, 'test.npy']), aug=False)
        self.target_val_list[dom] = iter(DataLoader(target_dataset_val,
                                                    batch_size=self.val_batch_size,
                                                    shuffle=False,
                                                    num_workers=config.args.worker,
                                                    pin_memory=True))

    '''
    Utility function to initialize pseudo data loader
    '''

    def initialize_pseudo_trgt_dataloader(self, dom):
        pseudo_target_dataset_train = PseudoTargetDataset('_'.join([dom, 'train.npy']),
                                                          self.pseudo_idx_list[dom],
                                                          self.pseudo_cls_label_list[dom])
        self.pseudo_target_dl_iter_train_list[dom] = iter(DataLoader(pseudo_target_dataset_train,
                                                                     batch_size=self.batch_size,
                                                                     shuffle=True,
                                                                     drop_last=True,
                                                                     num_workers=config.args.worker,
                                                                     pin_memory=True))
        print('Pseudo label set len: ', len(pseudo_target_dataset_train.aug_list))

    '''
    Target dataset validation
    '''

    def val_over_target_set(self):
        config.eval_time = time.time()
        self.get_all_val_target_dataloaders()
        with torch.no_grad():
            for dom in self.settings['trgt_datasets']:

                all_labels = []
                all_logits = [[] for _ in range(len(self.settings['src_datasets']))]
                target_val_list = self.target_val_list[dom]

                for data in tqdm(target_val_list, desc=dom):
                    _, images, label, _ = data

                    images = images.to(self.settings['device']).float()

                    feats = self.network.model['F'](self.network.model['B'](images))
                    logits = self.network.model['C'](feats, 'eval')

                    for i in range(len(self.settings['src_datasets'])):
                        all_logits[i].append(logits[i])
                    all_labels.append(label)

                domain_wise_pred_labels = []
                all_preds = 0
                for i in range(len(self.settings['src_datasets'])):
                    all_logits[i] = torch.cat(all_logits[i], dim=0).cpu()
                    all_preds += self.weight[i] * all_logits[i]
                    domain_wise_pred_labels.append(torch.argmax(all_logits[i], dim=1))
                all_labels = torch.cat(all_labels, dim=0).cpu()
                all_preds = torch.argmax(all_preds, dim=1)

                self.evaluation_results(all_labels, all_preds, domain_wise_pred_labels)

    '''
    Utility function to save experiment data
    '''

    def evaluation_results(self, true_labels, preds_labels, domain_wise_preds):

        for i in range(len(self.settings['src_datasets'])):
            acc = f1_score(true_labels, domain_wise_preds[i], average='micro')
            self.cal_metrics['domain_wise_acc'][i].append(acc)
            print(i, " acc: ", acc)

        micro_score = f1_score(true_labels, preds_labels, average='micro')
        macro_score = f1_score(true_labels, preds_labels, average='macro')
        met = [['Target macro score', macro_score],
               ['Target micro score(acc)', micro_score],
               ['Max accuracy', max(micro_score, self.max_acc)]]
        print(tabulate(met, headers=['metrics', 'value'], tablefmt="simple", showindex=True))

        self.cal_metrics['micro_score'].append(micro_score)
        self.cal_metrics['macro_score'].append(macro_score)
        self.max_acc = float(max(micro_score, self.max_acc))
        if self.current_iteration <= self.settings['enough_iter'] + 10:
            self.check_and_save_weights(micro_score)
        self.save_log()

        print('Time Cost: ', config.eval_time - config.begin_time)
        config.begin_time = time.time()

    '''
    Function to calculate the loss value
    '''

    def loss(self):

        tot_loss = 0
        logits = self.src_features['logits']
        labels = self.src_features['label']
        for i in range(len(self.src_data)):
            tot_loss += self.weight[i] * torch.nn.CrossEntropyLoss(reduction='mean')(logits[i], labels[i])
        if self.current_iteration > self.settings['enough_iter']:
            logits = self.pseudo_features['logits']
            pseudo_labels = self.pseudo_features['label']
            for i in range(len(self.src_data)):
                tot_loss += torch.nn.CrossEntropyLoss(reduction='mean')(logits[i], pseudo_labels[0].long())

        return tot_loss

    '''
    Function to implement the forward prop
    '''

    def forward(self):
        state = 'warm_up'
        image = []
        source_label = []
        for dom in self.src_data:
            image.append(self.src_data[dom]['images'])
            source_label.append(self.src_data[dom]['label'])

        if self.current_iteration > self.settings['enough_iter']:
            state = 'adapt'
            pseudo_label = []
            for dom in self.pseudo_data:
                image.append(self.pseudo_data[dom]['images'])
                pseudo_label.append(self.pseudo_data[dom]['label'])

        image = torch.cat(image, dim=0)

        feats = self.network.model['F'](self.network.model['B'](image))
        logits = self.network.model['C'](feats, state)

        if state == 'warm_up':
            self.src_features['logits'] = logits
            self.src_features['label'] = source_label
        else:
            self.src_features['logits'] = logits[0]
            self.src_features['label'] = source_label
            self.pseudo_features['logits'] = logits[1]
            self.pseudo_features['label'] = pseudo_label

    '''
    Function for training the the data
    This function is called at every iteration
    '''

    def train(self):
        for dom in self.settings['src_datasets']:
            try:
                self.src_data[dom] = {}
                _, self.src_data[dom]['images'], self.src_data[dom]['label'], self.src_data[dom]['domain_label'] = \
                    self.source_dl_iter_train_list[dom].next()
                self.src_data[dom]['images'] = \
                    Variable(self.src_data[dom]['images']).to(self.settings['device']).float()
                self.src_data[dom]['label'] = \
                    Variable(self.src_data[dom]['label']).to(self.settings['device']).long()
                self.src_data[dom]['domain_label'] = \
                    Variable(self.src_data[dom]['domain_label']).to(self.settings['device']).long()
            except StopIteration:
                self.initialize_src_train_dataloader(dom)
                self.src_data[dom] = {}
                _, self.src_data[dom]['images'], self.src_data[dom]['label'], self.src_data[dom]['domain_label'] = \
                    self.source_dl_iter_train_list[dom].next()
                self.src_data[dom]['images'] = \
                    Variable(self.src_data[dom]['images']).to(self.settings['device']).float()
                self.src_data[dom]['label'] = \
                    Variable(self.src_data[dom]['label']).to(self.settings['device']).long()
                self.src_data[dom]['domain_label'] = \
                    Variable(self.src_data[dom]['domain_label']).to(self.settings['device']).long()

        if self.current_iteration > self.settings['enough_iter']:
            if self.current_iteration == self.settings['enough_iter'] + 1:
                self.initialize_pseudo_target_indices()
                for dom in self.settings['trgt_datasets']:
                    self.initialize_pseudo_trgt_dataloader(dom)

            for dom in self.settings['trgt_datasets']:
                try:
                    self.pseudo_data[dom] = {}
                    _, self.pseudo_data[dom]['images'], self.pseudo_data[dom]['label'], _ = \
                        self.pseudo_target_dl_iter_train_list[dom].next()
                    self.pseudo_data[dom]['images'] = \
                        Variable(self.pseudo_data[dom]['images']).to(self.settings['device']).float()
                    self.pseudo_data[dom]['label'] = \
                        Variable(self.pseudo_data[dom]['label']).to(self.settings['device']).float()
                except StopIteration:
                    self.initialize_pseudo_target_indices()
                    self.initialize_pseudo_trgt_dataloader(dom)
                    self.pseudo_data[dom] = {}
                    _, self.pseudo_data[dom]['images'], self.pseudo_data[dom]['label'], _ = \
                        self.pseudo_target_dl_iter_train_list[dom].next()
                    self.pseudo_data[dom]['images'] = \
                        Variable(self.pseudo_data[dom]['images']).to(self.settings['device']).float()
                    self.pseudo_data[dom]['label'] = \
                        Variable(self.pseudo_data[dom]['label']).to(self.settings['device']).float()

        self.forward()
        self.optimizer.zero_grad()
        self.loss().backward()
        self.optimizer.step()


if __name__ == '__main__':
    raise NotImplementedError('Please check train file')
