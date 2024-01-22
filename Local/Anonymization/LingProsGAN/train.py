import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_data import *
from anonymization.WGAN.init_wgan import create_wgan
import logging
import torch
import yaml

logger = logging.getLogger(__name__)

def train_COVID_GAN(exp_root_path, path_to_exp_args, device='cuda'):

    path_to_exp_args = os.path.join(exp_root_path, path_to_exp_args)

    with open(path_to_exp_args, 'r') as file:
        exp_args = yaml.safe_load(file)

    # logger.info("Starting experiment")
    # Prepare data, generate manifest files (.json)
    aggregate_covid_data_for_GAN(
    exp_args['metadata_folder'],
    exp_args['metadata_filename'],
    exp_args['manifest_file_train'],
    exp_args['manifest_file_valid'],
    exp_args['manifest_file_test'],
    )

    # logger.info("Setting up dataloaders")
    # Set up datasets for training
    train_set = COVIDforGAN(exp_args['manifest_file_train'])

    gan_checkpoint = exp_args['model_parameters']
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=gan_checkpoint['batch_size'], shuffle=True)

    # logger.info("Start training")

    # Start training
    wgan = create_wgan(parameters=gan_checkpoint, device=device)
    if 'pretrained_gan' in exp_args:
        pt = torch.load(exp_args['pretrained_gan'], map_location='cuda')
        print('pretrained checkpoint is loaded.')
        wgan.G.load_state_dict(pt['generator_state_dict'])
        wgan.D.load_state_dict(pt['critic_state_dict'])

    wgan.train(train_loader, writer=None)
    # save checkpoint
    if not os.path.exists(os.path.dirname(exp_args['gan_save_path'])):
        os.mkdir(os.path.dirname(exp_args['gan_save_path']))
    torch.save(wgan.G.state_dict(), exp_args['gan_save_path'])
    torch.save(wgan.D.state_dict(), exp_args['critic_save_path'])


def main(exp_root_path, path_to_exp_args, device='cuda'):

    train_COVID_GAN(exp_root_path, path_to_exp_args, device)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
                    prog='COVID GAN',
                    description='train a GAN to generate fake COVID speaker embeddings',
                    )
    
    parser.add_argument('--exp_root_path')
    parser.add_argument('--logger_name', default='logger.log')
    parser.add_argument('--path_to_exp_args', default='args/exp_args.yml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    logger = logging.basicConfig(filename=os.path.join(args.exp_root_path, args.logger_name),
                    format='%(asctime)s:%(levelname)s:%(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filemode='w', 
                    level=logging.DEBUG)

    main(args.exp_root_path,
         args.path_to_exp_args, 
         args.device)

