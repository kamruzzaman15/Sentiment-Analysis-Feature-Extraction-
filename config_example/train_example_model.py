import os, argparse, uuid, json, torch

from training_utils import (
    add_ckpt_config,
    parameter_grid,
    path_relative_to_main,
    save_model_with_args,
)
from setup_logging import setup_logging
from example_trainer import ExampleTrainer

log, logpath = setup_logging()


def pretrain(params, ckpts, ckpt_configs):
    log.info('')
    log.info('Pretraining model...')

    for hyper in parameter_grid(params['hyper']):

        log.info('')
        log.info('Hyperparameters')

        for k, v in hyper.items():
            log.info('{}:\t{}'.format(k, v))

        for training in parameter_grid(params['training']):
            log.info('')
            log.info('Training parameters')

            for k, v in training.items():
                log.info('{}:\t{}'.format(k, v))

            trainer = ExampleTrainer(**hyper)

            log.info('')
            log.info('Training encoder...')
            log.info('')
            trained_model = trainer.fit(**training)

            log.info('')
            log.info('Saving model...')
            log.info('')

            model_info = { "model": trainer.model,
                           "hyper": hyper }
            ckpt_path = save_model_with_args(params, model_info, ckpts)
            log.info('ckpt path: ' + ckpt_path)
            log.info('')

            log.info('Saving checkpoint to configuration tree ({})...'.format(ckpt_configs))
            add_ckpt_config(ckpt_configs,
                            'test_model',
                            ckpt_path,
                            hyper,
                            training,
                            logpath)
            log.info('')


def main(args):
    with open(args.parameters) as f:
        params = json.load(f)

    if os.path.isabs(args.checkpoints):
        ckpt_dir = args.checkpoints
    else:
        ckpt_dir = path_relative_to_main(args.checkpoints)
    ckpt_configs = args.checkpoint_config_path

    pretrain(params[args.encodertype],
             ckpt_dir,
             ckpt_configs)
                                       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain type encoders')
    parser.add_argument('--parameters',
                        type=str,
                        default='pretrain_type_encoder.json',
                        help='Path to a JSON containing parameters to sweep')
    parser.add_argument('--checkpoints',
                        type=str,
                        default='checkpoints/',
                        help='Where to save the models to. If not an absolute path, the path is relative to main script file.')
    parser.add_argument('--checkpoint_config_path',
                        type=str,
                        default='checkpoint_config_tree.json',
                        help='Path to a JSON containing mapping between the checkpoint paths and the associated configurations. Path relative to the calling directory.')
    args = parser.parse_args()

    main(args)
