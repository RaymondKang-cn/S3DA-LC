import numpy as np
import torch
import config as config
from trainer_supervised import TrainerG


def main():
    trainer = TrainerG()

    enough_iter = config.settings['enough_iter']
    max_iter = config.settings['max_iter']
    val_after = config.settings['val_after']
    batch_size = config.settings['batch_size']
    val_batch_size = int(config.settings['val_batch_size_factor'] * config.settings['batch_size'])

    print('enough_iter:{}\nmax_iter:{}\nval_after:{}\ntraining batch_size for each domain :{}\nval_batch_size:{}\n'
          .format(enough_iter, max_iter, val_after, batch_size, val_batch_size))

    if config.settings['load_model']:
        print('Resuming training from iteration :{}'.format(config.settings['model_dict']['iter']))
        trainer.current_iteration = config.settings['model_dict']['iter']
        trainer.load_model()
        trainer.load_optimizers()

    while True:

        

        if trainer.current_iteration % val_after == 0:
            print("\n----------- Evaluate at " + str(trainer.current_iteration) + ' -----------')
            print('From train: torch_seed={} numpy seed={}'.format(torch.initial_seed(), np.random.get_state()[1][0]))
            print('Running dataset = {}, task = {}, net = {}'.format(
                config.dataset, config.task, config.args.net))
            print('Parameter tau = {}, fix = {}, alpha = {}'.format(
                config.args.tau, config.args.fix, config.args.alpha))
            trainer.network.eval()
            trainer.val_over_target_set()

        trainer.network.train()
        trainer.train()
        
        if trainer.current_iteration > config.settings['max_iter']:
            break

        trainer.current_iteration += 1


if __name__ == '__main__':
    main()
