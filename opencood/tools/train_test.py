
import opencood.hypes_yaml.yaml_utils as yaml_utils
import argparse
import wandb
import yaml
def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    # parser.add_argument('--model_dir', default='',
    #                     help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt, unknown = parser.parse_known_args()
    return opt, unknown

def update_yaml_with_args(hypes, params_dict):
    for key, value in params_dict.items():
        keys = key.split('.')
        d = hypes
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return hypes

def param_to_dict(param:list):
    param_dict = {}
    for p in param:
        key, value = p.lstrip('--').split('=', 1)
        value_dict = yaml.safe_load(value)
        param_dict[key] = value_dict
    return param_dict

def main():
    opt, unknown_params = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print(f'Loaded hypes: {hypes}')

    unknown_params_dict = param_to_dict(unknown_params)
    print(f'==============Unknown params: {unknown_params_dict}')
    wandb.init(project="rinne-former", config=unknown_params_dict)
    config = wandb.config
    print(config)
    hypes = update_yaml_with_args(hypes, config)
    print(f'Updated hypes: {hypes}')
    

    
if __name__ == '__main__':
    main()