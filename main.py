import os
import torch
import numpy as np
import random
from utils import create_logger,AverageMeter,mkdir,save_to_txt
import matplotlib.pyplot as plt
import time
from Config import get_args
from load_data import load_data
from Model import HTLNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def set_random_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)


def get_optimizer(model,args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.Adam(params,lr=initial_lr,weight_decay=args.weight_decay)
    return optimizer

def get_scheduler(optimizer,args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler

def test(model,target_test_loader,args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        ground_true = []
        predict_label = []
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model.predict(data)
            loss = criterion(output,target)
            test_loss.update(loss.item())
            ground_true.append(target.cpu().detach().numpy())
            predict_label.append(output.cpu().detach().numpy())
    return test_loss.avg, np.concatenate(ground_true), np.concatenate(predict_label)


def train(source_loader,target_train_loader,target_valid_loader, target_test_loader,model,optimizer,lr_scheduler,args):
    if args.is_save_logging:
        mkdir(args.save_root)
        log_name = args.save_root + '/train info.log'
        log, consoleHander, fileHander= create_logger(filename=log_name)
        log.critical(args)
    else:
        log, consoleHander = create_logger()
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = max(len_source_loader,len_target_loader)

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    stop = 0
    min_test_loss = 10
    last_best_model = None
    for e in range(1,args.n_epoch+1):
        model.train()
        pred_loss, mmd_loss, ca_loss = AverageMeter(),AverageMeter(),AverageMeter()
        total_loss = AverageMeter()
        sub_mmd = AverageMeter()
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        for iter_num in range(1,n_batch+1):
            if iter_num % len_source_loader == 0:
                iter_source = iter(source_loader)
            if iter_num % len_target_loader == 0:
                iter_target = iter(target_train_loader)

            data_source, label_source = next(iter_source)
            data_target, _ = next(iter_target)
            data_source = data_source.to(args.device)
            label_source = label_source.to(args.device)
            data_target = data_target.to(args.device)

            l_pred, l_mmd, l_ca, mmd = model(data_source,data_target,label_source)

            loss = l_pred + args.beta*l_ca + args.alpha*mmd


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            pred_loss.update(l_pred.item())
            mmd_loss.update(l_mmd.item())
            ca_loss.update(l_ca.item())
            sub_mmd.update(mmd.item())
            total_loss.update(loss.item())
        train_info = f'Epoch:[{e}/{args.n_epoch}], pred_loss:{pred_loss.avg:.4f}, ' \
                     f'mmd_loss:{sub_mmd.avg:.4f}->{args.alpha*sub_mmd.avg:.4f}, ' \
                     f'ca_loss:{ca_loss.avg:.4f}->{args.beta*ca_loss.avg:.4f}, ' \
                     f'total_loss:{total_loss.avg:.4f}'
        log.info(train_info)

        ##################### test #######################
        if target_valid_loader is None:
            target_valid_loader = target_test_loader
        stop += 1
        #### valid
        valid_loss, _, _ = test(model, target_valid_loader, args)
        valid_info = f"valid_loss:{valid_loss:.4f}  lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
        log.warning(valid_info)

        if min_test_loss > valid_loss:
            min_test_loss = valid_loss

            test_loss, true_label, pred_label = test(model, target_test_loader, args)
            test_info = f"Epoch:[{e}/{args.n_epoch}],  test_loss:{test_loss:.4f}"
            log.error(test_info)

            stop = 0
            #######plot test results#########
            if args.is_plot_test_results:
                plt.plot(true_label, label='true')
                plt.plot(pred_label, label='pred')
                plt.title(f"Epoch:{e}, test loss:{test_loss:.4f}")
                plt.legend()
                plt.show()
            ####### save model ########
            if args.is_save_best_model:
                if last_best_model is not None:
                    os.remove(last_best_model)  # delete last best model

                save_folder = args.save_root + '/pth'
                mkdir(save_folder)
                best_model = os.path.join(save_folder, f'Epoch{e}.pth')
                torch.save(model.state_dict(), best_model)
                last_best_model = best_model
            #########save test results (test info) to txt #####
            if args.is_save_to_txt:
                txt_path = args.save_root + '/test_info.txt'
                time_now = time.strftime("%Y-%m-%d", time.localtime())
                info = time_now + f' {args.alpha}-{args.beta}, epoch = {e}, test_loss:{test_loss:.6f}'
                save_to_txt(txt_path,info)
            #########save test results (predict value) to np ######
            if args.is_save_test_results:
                np.save(args.save_root+'/pred_label',pred_label)
                np.save(args.save_root+'/true_label',true_label)
        if args.early_stop > 0 and stop > args.early_stop:
            print(' Early Stop !')
            if args.is_save_logging:
                log.removeHandler(consoleHander)
                log.removeHandler(fileHander)
            else:
                log.removeHandler(consoleHander)
            break
    if args.is_save_logging:
        log.removeHandler(consoleHander)
        log.removeHandler(fileHander)
    else:
        log.removeHandler(consoleHander)

def main(args):
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_valid_loader, target_test_loader = load_data(args)
    model = HTLNet(args).to(args.device)
    optimizer = get_optimizer(model, args)
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_valid_loader, target_test_loader, model, optimizer, scheduler, args)
    torch.cuda.empty_cache()
    del source_loader
    del target_train_loader
    del target_valid_loader
    del target_test_loader

def main_experiment():
    args = get_args()
    test_battery_id = [1, 2, 3, 4, 5]
    for i in range(5):  # 进行5次实验
        for id in test_battery_id:
            try:
                print(f'experiment--{i}--{id}')
                setattr(args, 'test_battery_id', id)
                save_root = f'results_ours/test_battery-{id}/experiment-{i + 1}'
                setattr(args, 'save_root', save_root)
                main(args)
            except:
                continue


if __name__ == '__main__':
    main_experiment()



