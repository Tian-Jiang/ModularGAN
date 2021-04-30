import os
import argparse
from trainer import Trainer
from torch.backends import cudnn
from celeba_loader import data_loader

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # for fast training
    cudnn.benchmark = True
    

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    celeba_loader = data_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                config.crop_size, config.image_size, config.batchsz, config.mode)

    trainer = Trainer(celeba_loader, config)

    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'test':
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # configuration used in models
    parser.add_argument('--c64', type=int, default=64, help='the starting dim of conv layers')
    parser.add_argument('--rb6', type=int, default=6, help='layer number of residual blocks')
    parser.add_argument('--c256', type=int, default=256, help='the starting dim of conv layers')
    parser.add_argument('--c2048', type=int, default=2048, help='the starting dim of deconv layers')
    parser.add_argument('--attr_dim', type=int, default=1, help='layer number of gender or smailing')
    parser.add_argument('--hair_dim', type=int, default=3, help='layer number of gender or smailing')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_cyc', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # training configuration
    parser.add_argument('--batchsz', type=int, default=16, help="batch size")
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each ETR update')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate of G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate of D')
    parser.add_argument('--r_lr', type=float, default=0.0001, help='learning rate of Reconstructor')
    parser.add_argument('--t_lr', type=float, default=0.0001, help='learning rate of Transformer')
    parser.add_argument('--e_lr', type=float, default=0.0001, help='learning rate of Encoder')
    parser.add_argument('--decay_rate', type=int, default=100, help='rate for decaying lr')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--crop_size', type=int, default=178, help='crop size')
    parser.add_argument('--image_size', type=int, default=128, help='image size')
    parser.add_argument('--selected_attrs', '--list', help='selected attributes from the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Smiling'])
    parser.add_argument('--train_iters', type=int, default=200000, help='how many times to train')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')

    # test configuration
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # miscellaneous
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # directories
    parser.add_argument('--celeba_image_dir', type=str, default='/content/datasets/celeba/img_align_celeba')
    parser.add_argument('--attr_path', type=str, default='/content/datasets/celeba/list_attr_celeba.txt')
    parser.add_argument('--log_dir', type=str, default='/content/drive/MyDrive/ModularGAN/modularGAN/logs')
    parser.add_argument('--model_save_dir', type=str, default='/content/drive/MyDrive/ModularGAN/modularGAN/models')
    parser.add_argument('--sample_dir', type=str, default='/content/drive/MyDrive/ModularGAN/modularGAN/samples')
    parser.add_argument('--result_dir', type=str, default='/content/drive/MyDrive/ModularGAN/modularGAN/results')

    # step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
