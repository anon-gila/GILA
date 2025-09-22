

import argparse
import sys
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict

from core.model_handler import ModelHandler
from sklearn.model_selection import StratifiedKFold

from kfold_data import load_data

################################################################################
# Main #
################################################################################



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config, args):


    random_seed = [args['seed1']]
    
    X,y = load_data(config['dataset_name'])
    
    acc_5_list, auc_5_list, prec_5_list, rec_5_list, f1_5_list, adj_homo_list =[],[],[],[],[],[]
    
    for seed in random_seed:

        acc_list, prec_list, rec_list, f1_list, auc_list =[],[],[],[],[]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            
            print(f"🌸🌸🌸🌸🌸🌸🌸🌸🌸 {config['dataset_name']},{fold+1}fold!! 🌸🌸🌸🌸🌸🌸🌸🌸🌸")
            set_random_seed(seed)
            data_index = (train_index, test_index)
            model = ModelHandler(config, args, data_index)
            

            model.train(fold)
            acc,auc,precision, recall, f1, adj_homo = model.test(args, fold)
            acc_list.append(acc)
            auc_list.append(auc)
            prec_list.append(precision)
            rec_list.append(recall)
            f1_list.append(f1)
            adj_homo_list.append(adj_homo)
            
            
        import datetime
        from pytz import timezone   
        now = datetime.datetime.now(timezone('Asia/Seoul'))
        
        dataname = cfg['config'].strip().split('/')[1]

       
        result = [now, dataname, np.mean(auc_list),np.std(auc_list),np.mean(f1_list),np.std(f1_list)]
        
        ratio = cfg['ratio']
       
        import csv
        with open(f'results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(result)
            
                # print(acc,auc,precision, recall, f1)
        acc_5_list.append(np.mean(acc_list))
        auc_5_list.append(np.mean(auc_list))
        rec_5_list.append(np.mean(rec_list))
        f1_5_list.append(np.mean(f1_list))
        prec_5_list.append(np.mean(prec_list))
    

################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(arg,config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.full_load(setting)

        

    return config

def get_args(num):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('-config', default='config/cleveland-0_vs_4/idgl.yml', type=str, help='path to the config file')
    parser.add_argument('-multi_run', default=False, help='flag: multi run')
    parser.add_argument('-ratio', default=1, type=float)
    parser.add_argument('-fake_samples', default=True, type=bool)  ####### helper nodes
    parser.add_argument('-smote', default=False, type=bool)
    parser.add_argument('-fake_rate', default=1, type=float) ## helper nodes ratio
    parser.add_argument('-updaters', default=True, type=bool) ####### updater 
    parser.add_argument('-data_seed', default=num, type=int) 
    parser.add_argument('-knn_size', default=10, type=int)
    parser.add_argument('-noise_rate', default=0, type=float) 
    parser.add_argument('-seed1', default=45, type=int) 
    
    

    
    args = parser.parse_args()
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    cfg = get_args(0)
    cfg = vars(cfg)
    config = get_config(cfg, cfg['config'])
    main(config, cfg)

    

