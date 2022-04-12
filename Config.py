import argparse

__all__ = ['get_args']

def get_args():
    parser = argparse.ArgumentParser(description='HTL for battery capacity estimation !')
    parser.add_argument('--input_channel',default=3)
    parser.add_argument('--embedding_length',default=128)
    parser.add_argument('--use_dsbn',type=bool,default=False)


    parser.add_argument('--source_dir',type=str, default='data/MIT2_2')
    parser.add_argument('--target_dir',type=str, default='data/MIT2_6')
    parser.add_argument('--test_battery_id', type=int, default='5',help='choice from the range of [1,5]')



    parser.add_argument('--ca_loss_type',default='L1',type=str,choices=['L1','L2'])

    parser.add_argument('--normalize_type',default='minmax',choices=['minmax','standerd'])
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--seed',default=2022)

    # L = pred_loss + alpha*domain_loss + beta*ca_loss
    parser.add_argument('--alpha',type=float,default=0.05)
    parser.add_argument('--beta',type=float,default=0.1)

    # optimizer
    parser.add_argument('--lr',type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=bool, default=True)

    parser.add_argument('--device',default='cuda')
    parser.add_argument('--n_epoch',default=50)
    parser.add_argument('--early_stop', default=20)

    # reulsts
    parser.add_argument('--is_plot_test_results',default=False)
    parser.add_argument('--is_save_logging',default=False)
    parser.add_argument('--is_save_best_model',default=False)
    parser.add_argument('--is_save_to_txt',default=False)
    parser.add_argument('--is_save_test_results',default=False)

    parser.add_argument('--save_root',default='results')

    args = parser.parse_args()
    return args