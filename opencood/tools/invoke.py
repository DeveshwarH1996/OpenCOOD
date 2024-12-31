import argparse
import subprocess
import wandb
import yaml

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    opt, unknown = parser.parse_known_args()
    return opt, unknown

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def update_yaml_with_args(hypes, params):
    for param in params:
        key, value = param.split('=', 1)
        value_dict = yaml.safe_load(value)
        keys = key.split('.')
        d = hypes
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value_dict
    return hypes

def config_to_dict(config):
    """
    Convert wandb.config object to a dictionary.
    """
    return {k: v for k, v in config.items()}

def main():
    opt, unknown_params = train_parser()
    hypes_yaml = 'wandb/hypes_yaml/point_pillar_v2xvit_PointTransformer_multi_head.yaml'
    hypes = load_yaml(hypes_yaml)
    print(hypes)
    # Initialize wandb
    wandb.init(project="rinne-former", config=hypes)
    config = wandb.config
    print(config)
    # Update hypes with command line arguments
    hypes = update_yaml_with_args(hypes, unknown_params)
    print(hypes)
    config_dict = config_to_dict(config)
    
    # Update yaml file with wandb config
    save_yaml(config_dict, hypes_yaml)

    # Invoke train.py with updated yaml file
    command = [
        "CUDA_VISIBLE_DEVICES=0,1",
        "python", "-m", "torch.distributed.launch",
        "--nproc_per_node=2", "--use_env",
        "opencood/tools/train.py",
        "--hypes_yaml", hypes_yaml
    ]
    subprocess.run(" ".join(command), shell=True)

if __name__ == '__main__':
    main()