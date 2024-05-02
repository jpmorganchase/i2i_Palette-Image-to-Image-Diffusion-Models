import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp
import copy 

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    networks[0].forget_alpha = opt['forget_alpha']
    networks[0].max_loss = opt['max_loss']
    networks[0].learn_noise = opt['learn_noise']
    networks[0].learn_others = opt['learn_others']

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    tmp_ckpt = torch.load(opt['ckpt'])
    new_ckpt={}
    for keyname in tmp_ckpt:
        if 'teacher_denoise_fn' not in keyname:
            new_ckpt['module.'+keyname] = tmp_ckpt[keyname]
    status = model.netG.load_state_dict(new_ckpt)
    # print(model.netG)
    print(status)
    
    if opt['fix_decoder'] or opt['learn_others']:
        print('Fixing decoder!!!!!!!!!!!!!!!!! or using learn_others')
        model.netG.module.teacher_denoise_fn = copy.deepcopy(model.netG.module.denoise_fn)
    # model.netG.module.forget_alpha = opt.forget_alpha

    # new_ckpt={}
    # for keyname in tmp_ckpt:
    #     if 'teacher_denoise_fn' not in keyname:
    #         new_ckpt['module.'+keyname] = tmp_ckpt[keyname]
    # status = model.netG.load_state_dict(new_ckpt, strict=False)
    # print(status)

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train' or opt['phase'] == 'unlearn':
            if opt['fix_decoder']:
                model.unlearn_fix_decoder()
            else:
                model.unlearn()
        else:
            model.test()
    finally:
        phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('--fix_decoder', default=0, type=int)
    parser.add_argument('--forget_alpha', default=0.1, type=float)
    parser.add_argument('--max_loss', default=0, type=int)
    parser.add_argument('--learn_noise', default=0, type=int)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--learn_others', default=0, type=int)

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    opt['fix_decoder']=args.fix_decoder
    opt['forget_alpha']=args.forget_alpha
    opt['max_loss']=args.max_loss
    opt['learn_noise']=args.learn_noise
    opt['ckpt']=args.ckpt
    opt['learn_others']=args.learn_others

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)