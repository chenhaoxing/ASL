import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import math
from tqdm import tqdm
from pandas import DataFrame
import sys 
sys.path.append("..") 
sys.path.append("../..") 



from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

from model import ASL
from utils import get_dataset, get_proto_accuracy
from global_utils import Averager, Averager_with_interval, get_outputs_c_h, get_inputs_and_outputs, get_semantic_size


def save_model(model, args, tag):
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, tag]) + '.pt'))
    if args.multi_gpu:
        model = model.module
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

def save_checkpoint(args, model, train_log, optimizer, global_task_count, tag):
    if args.multi_gpu:
        model = model.module
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'train_log': train_log,
        'val_acc': train_log['max_acc'],
        'optimizer': optimizer.state_dict(),
        'global_task_count': global_task_count
    }
    checkpoint_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, tag]) + '_checkpoint.pt.tar'))
    with open(checkpoint_path, 'wb') as f:
        torch.save(state, f)

if __name__ == '__main__':
    import argparse

    # should be updated in different models
    parser = argparse.ArgumentParser('Prototypical Networks with ASL')
    parser.add_argument('--model-name', type=str, default='protonet_asl',
        help='Name of the model.')

    # experimental settings
    parser.add_argument('--data-folder', type=str, default='../../datasets', 
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--train-data', type=str, default='cub', 
        choices=['cub', 'sun'],
        help='Name of the dataset used in meta-train phase.')
    parser.add_argument('--test-data', type=str, default='cub', 
        choices=['cub', 'sun'],
        help='Name of the dataset used in meta-test phase.')
    parser.add_argument('--backbone', type=str, default='conv4', 
        choices=['conv4', 'resnet12'],
        help='Name of the CNN backbone.')
    parser.add_argument('--lr', type=float, default=0.001,
        help='Initial learning rate (default: 0.001).')

    parser.add_argument('--num-shots', type=int, default=5, choices=[1, 5, 10],
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, choices=[5, 20], 
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default=15,
        help='Number of query examples per class (k in "k-shot", default: 15).')

    parser.add_argument('--batch-tasks', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')

    # follows Closer_Look, 40000 for 5 shots and 60000 for 1 shot
    parser.add_argument('--train-tasks', type=int, default=40000,
        help='Number of tasks in the training phase (default: 40000).')
    parser.add_argument('--val-tasks', type=int, default=600,
        help='Number of tasks in the validation phase (default: 600).')
    parser.add_argument('--test-tasks', type=int, default=10000,
        help='Number of tasks in the testing phase (default: 10000).')

    parser.add_argument('--augment', type=bool, default=True,
        help='Augment the training dataset (default: True).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15000, 30000, 45000, 60000], 
        help='Decrease learning rate at these number of tasks.')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='Learning rate decreasing ratio (default: 0.1).')

    parser.add_argument('--valid-every-tasks', type=int, default=1000,
        help='Number of tasks for each validation (default: 1000).')

    # arguments of program
    parser.add_argument('--num-workers', type=int, default=2,
        help='Number of workers for data loading (default: 2).')
    parser.add_argument('--download', action='store_true',
        help='Download the dataset in the data folder.')
    parser.add_argument('--use-cuda', type=bool, default=True,
        help='Use CUDA if available.')
    parser.add_argument('--multi-gpu', action='store_true',
        help='True if use multiple GPUs. Else, use single GPU.')

    # arguments for resume (i.e. checkpoint)
    parser.add_argument('--resume', action='store_true',
        help='If training starts from resume.')
    parser.add_argument('--resume-folder', type=str, default=None,
        help='Path to the folder the resume is saved to.')

    # special arguments for ASL
    parser.add_argument('--alpha', type=float, default=1.0,
        help='Value of the trade-off parameter of asl in loss function(default: 1.0).')

    # arguments of semantic
    parser.add_argument('--semantic-type', type=str, nargs='+',
        choices=['class_attributes', 'image_attributes'],
        help='Semantic type.')

    args = parser.parse_args()

    # make folder and tensorboard writer to save model and results
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.record_folder = './{}_{}_{}_{}_{}'.format(args.train_data, args.test_data, args.model_name, args.backbone, cur_time)
    # writer = SummaryWriter(args.record_folder)
    os.makedirs(args.record_folder, exist_ok=True)

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    elif args.use_cuda:
        raise RuntimeError('You are using GPU mode, but GPUs are not available!')

    # construct model and optimizer
    assert (args.train_data == args.test_data)
    args.image_len = 84
    args.semantic_size = get_semantic_size(args)
    args.out_channels, args.feature_h = get_outputs_c_h(args.backbone, args.image_len)

    model = ASL(args.backbone, args.semantic_size, args.out_channels)

    if args.use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        num_gpus = torch.cuda.device_count()
        if args.multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()
    crition = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    # training from the checkpoint
    if args.resume and args.resume_folder is not None:
        # load checkpoint
        checkpoint_path = os.path.join(args.resume_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, 'max_acc']) + '_checkpoint.pt.tar'))    # tag='max_acc' can be changed
        state = torch.load(checkpoint_path)
        resumed_state = state['state_dict']
        if args.multi_gpu:
            model.module.load_state_dict(resumed_state)
        else:
            model.load_state_dict(resumed_state)
        train_log = state['train_log']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_task_count = state['global_task_count']

        print('global_task_count: {}, initial_lr: {}'.format(str(global_task_count), str(initial_lr)))

    # training from scratch
    else:
        train_log = {}
        train_log['args'] = vars(args)
        train_log['train_loss'] = []
        train_log['train_acc'] = []
        train_log['val_loss'] = []
        train_log['val_acc'] = []
        train_log['max_acc'] = 0.0
        train_log['max_acc_i_task'] = 0
        initial_lr = args.lr
        global_task_count = 0

    # save the args into .json file
    with open(os.path.join(args.record_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # get datasets and dataloaders
    train_dataset = get_dataset(args, dataset_name=args.train_data, phase='train')
    val_dataset = get_dataset(args, dataset_name=args.test_data, phase='val')
    test_dataset = get_dataset(args, dataset_name=args.test_data, phase='test')

    train_loader = BatchMetaDataLoader(train_dataset,
                                     batch_size=args.batch_tasks,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

    val_loader = BatchMetaDataLoader(val_dataset,
                                     batch_size=args.batch_tasks,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

    test_loader = BatchMetaDataLoader(test_dataset,
                                     batch_size=args.batch_tasks,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True)
    
    # training
    with tqdm(train_loader, total=int(args.train_tasks/args.batch_tasks), initial=int(global_task_count / args.batch_tasks)) as pbar:

        for i_train_batch, train_batch in enumerate(pbar, int(global_task_count / args.batch_tasks)+1):

            if i_train_batch > (args.train_tasks / args.batch_tasks):
                break

            model.train()

            # check if lr should decrease as in schedule
            if (i_train_batch * args.batch_tasks) in args.schedule:
                initial_lr *= args.gamma
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr

            global_task_count += args.batch_tasks

            support_inputs, support_targets, support_semantics = get_inputs_and_outputs(args, train_batch['train'])
            query_inputs, query_targets, query_semantics = get_inputs_and_outputs(args, train_batch['test'])

            support_embeddings, attr_s, attr_label_s = model(support_inputs, semantics=support_semantics,
                                                         Support=True)
            query_embeddings, attr_q, attr_label_q = model(query_inputs, semantics=query_semantics)

            prototypes = get_prototypes(support_embeddings, support_targets,
                                        train_dataset.num_classes_per_task)

            addition_loss_s = crition(attr_s, attr_label_s)
            addition_loss_q = crition(attr_q, attr_label_q)

            train_loss = prototypical_loss(prototypes, query_embeddings, query_targets) + args.alpha * (addition_loss_s + addition_loss_q)
            train_acc = get_proto_accuracy(prototypes, query_embeddings, query_targets)
            del attr_s, attr_label_s, attr_q, attr_label_q
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pbar.set_postfix(train_acc='{0:.4f}'.format(train_acc.item()))

            # validation
            if global_task_count % args.valid_every_tasks == 0:
                val_loss_averager = Averager()
                val_acc_averager = Averager_with_interval()

                model.eval()
                with torch.no_grad():
                    for i_val_batch, val_batch in enumerate(val_loader, 1):

                        if i_val_batch > (args.val_tasks / args.batch_tasks):
                            break

                        support_inputs, support_targets, support_semantics = get_inputs_and_outputs(args, val_batch['train'])
                        query_inputs, query_targets, query_semantics = get_inputs_and_outputs(args, val_batch['test'])

                        support_embeddings, attr_s, attr_label_s = model(support_inputs, semantics=support_semantics,
                                                                     Support=True)
                        query_embeddings, attr_q, attr_label_q = model(query_inputs, semantics=query_semantics)

                        prototypes = get_prototypes(support_embeddings, support_targets,
                                                    val_dataset.num_classes_per_task)
                        addition_loss_s = crition(attr_s, attr_label_s)
                        addition_loss_q = crition(attr_q, attr_label_q)

                        val_loss = prototypical_loss(prototypes, query_embeddings, query_targets) + args.alpha * (
                                    addition_loss_s + addition_loss_q)
                        val_acc = get_proto_accuracy(prototypes, query_embeddings, query_targets)
                        del attr_s, attr_label_s, attr_q, attr_label_q
                        val_loss_averager.add(val_loss.item())
                        val_acc_averager.add(val_acc.item())

                # record
                val_acc_mean = val_acc_averager.item()
                print('global_task_count: {}, val_acc_mean: {}'.format(str(global_task_count), str(val_acc_mean)))
                
                if val_acc_mean > train_log['max_acc']:
                    train_log['max_acc'] = val_acc_mean
                    train_log['max_acc_i_task'] = global_task_count
                    save_model(model, args, tag='max_acc')

                train_log['train_loss'].append(train_loss.item())
                train_log['train_acc'].append(train_acc.item())
                train_log['val_loss'].append(val_loss_averager.item())
                train_log['val_acc'].append(val_acc_mean)

                save_checkpoint(args, model, train_log, optimizer, global_task_count, tag='max_acc')
                del val_loss_averager, val_acc_averager


    # testing
    test_loss_averager = Averager()
    test_acc_averager = Averager_with_interval()
    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, total=int(args.test_tasks/args.batch_tasks)) as pbar:
            for i_test_batch, test_batch in enumerate(pbar, 1): 

                if i_test_batch > (args.test_tasks / args.batch_tasks):
                    break

                support_inputs, support_targets, support_semantics = get_inputs_and_outputs(args, test_batch['train'])
                query_inputs, query_targets, query_semantics = get_inputs_and_outputs(args, test_batch['test'])

                support_embeddings, attr_s, attr_label_s = model(support_inputs, semantics=support_semantics,
                                                             Support=True)
                query_embeddings, attr_q, attr_label_q = model(query_inputs, semantics=query_semantics)

                prototypes = get_prototypes(support_embeddings, support_targets, 
                                            test_dataset.num_classes_per_task)

                addition_loss_s = crition(attr_s, attr_label_s)
                addition_loss_q = crition(attr_q, attr_label_q)

                test_loss = prototypical_loss(prototypes, query_embeddings, query_targets) + args.alpha * (
                        addition_loss_s + addition_loss_q)
                test_acc = get_proto_accuracy(prototypes, query_embeddings, query_targets)
                del attr_s, attr_label_s, attr_q, attr_label_q
                pbar.set_postfix(test_acc='{0:.4f}'.format(test_acc.item()))

                test_loss_averager.add(test_loss.item())
                test_acc_averager.add(test_acc.item())

        # record
        index_values = [
            'test_acc',
            'best_i_task',    # the best_i_task of the highest val_acc
            'best_train_acc',    # the train_acc according to the best_i_task of the highest val_acc
            'best_train_loss',    # the train_loss according to the best_i_task of the highest val_acc
            'best_val_acc',
            'best_val_loss'
        ]
        print(test_acc_averager.item(return_str=True))
        best_index = int(train_log['max_acc_i_task'] / args.valid_every_tasks) - 1
        test_record = {}
        test_record_data = [
            test_acc_averager.item(return_str=True),
            str(train_log['max_acc_i_task']),
            str(train_log['train_acc'][best_index]),
            str(train_log['train_loss'][best_index]),
            str(train_log['max_acc']),
            str(train_log['val_loss'][best_index]),
        ]
        test_record[args.record_folder] = test_record_data
        test_record_file = os.path.join(args.record_folder, 'record_{}_{}_{}shot.csv'.format(args.train_data, args.test_data, args.num_shots))
        DataFrame(test_record, index=index_values).to_csv(test_record_file)

