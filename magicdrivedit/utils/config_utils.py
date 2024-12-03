import argparse
import json
import os
import datetime
from glob import glob

from hydra import compose, initialize
from omegaconf import OmegaConf
from mmengine.config import Config, DictAction


def parse_args(training=False):
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    # ======================================================
    # General
    # ======================================================
    parser.add_argument(
        "--ckpt-path",
        default=None,
        type=str,
        help="path to model ckpt; will overwrite cfg.model.from_pretrained if specified",
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None

    cfg_options = args.cfg_options
    args.cfg_options = None
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    # make to the last, has the highest priority
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    return cfg


def read_config(config_path):
    cfg = Config.fromfile(config_path)
    return cfg


def parse_configs(training=False):
    args = parse_args(training)
    cfg = read_config(args.config)
    cfg = merge_args(cfg, args, training)
    return cfg


def define_experiment_workspace(cfg, get_last_workspace=False, use_date=False):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(cfg.outputs, exist_ok=True)
    tag = cfg.get("tag", "")

    # Create an experiment folder
    model_name = cfg.model["type"].replace("/", "-")
    cfg_name = os.path.splitext(os.path.basename(cfg.config))[0]
    cfg_name = cfg_name if tag == "" else f"{cfg_name}_{tag}"
    if use_date:
        date_suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        exp_name = f"{model_name}_{cfg_name}_{date_suffix}"
    else:
        experiment_index = len(glob(f"{cfg.outputs}/*"))
        if get_last_workspace:
            experiment_index -= 1
        exp_name = f"{experiment_index:03d}-{model_name}_{cfg_name}"
    exp_dir = f"{cfg.outputs}/{exp_name}"
    return exp_name, exp_dir


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def mmengine_conf_get(cfg, key, default=None):
    m = key
    new_item = cfg
    while "." in m:
        p, m = m.split(".", 1)
        if new_item is None:
            return default
        new_item = new_item.get(p)
    return new_item.get(m, default)


def mmengine_conf_set(cfg, key, value):
    p, m = key.rsplit(".", 1)
    new_item = mmengine_conf_get(cfg, p)
    if new_item is None:
        raise KeyError("we cannot add key!")
    old_value = new_item.get(m)
    assert old_value is None or value is None or old_value.__class__ == value.__class__
    setattr(new_item, m, value)


def set_omegaconf_key_value(cfg, key, value):
    p, m = key.rsplit(".", 1)
    node = cfg
    for pk in p.split("."):
        node = getattr(node, pk)
    node[m] = value


def merge_dataset_cfg(cfg, data_cfg_name, data_cfg_overrides, num_frames=None):
    overrides = [f'+dataset={data_cfg_name}']
    if num_frames is not None:
        overrides.append(f'+model.video_length={num_frames}')
    data_cfg_overrides_rest = []
    for (k, v) in data_cfg_overrides:
        if k.startswith("+"):
            overrides.append(f'{k}={v}')
        else:
            data_cfg_overrides_rest.append((k, v))
    with initialize(version_base=None, config_path="../../configs"):
        dataset_cfg = compose(overrides=overrides)
    for (k, v) in data_cfg_overrides_rest:
        set_omegaconf_key_value(dataset_cfg, k, v)
    # train
    dataset = Config(OmegaConf.to_container(
        dataset_cfg.dataset.data.train, resolve=True))
    # set img_collate_param
    dataset.img_collate_param = cfg.img_collate_param_train
    dataset.img_collate_param.template = dataset_cfg.dataset.template
    # val
    val_dataset = Config(OmegaConf.to_container(
        dataset_cfg.dataset.data.val, resolve=True))
    # set img_collate_param
    val_dataset.img_collate_param = cfg.img_collate_param_train
    val_dataset.img_collate_param.template = dataset_cfg.dataset.template
    val_dataset.img_collate_param.is_train = False  # Important!
    return dataset.to_dict(), val_dataset.to_dict()
