import json, os, uuid, torch, inspect, sys, time, datetime
import logging
from itertools import product
import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from filelock import FileLock
from collections import defaultdict

log = logging.getLogger(__name__)

def parameter_grid(param_dict):
    ks = list(param_dict.keys())
    vlists = []
    for k, v in param_dict.items():
        if isinstance(v, dict):
            vlists.append(parameter_grid(v))
        elif isinstance(v, list):
            vlists.append(v)
        else:
            errmsg = ("param_dict must be a dictionary contining lists or "
                      "recursively other param_dicts")
            raise ValueError(errmsg)
    for configuration in product(*vlists):
        yield dict(zip(ks, configuration))

def save_model(data_dict, ckpt_dir):
    rand_hash = uuid.uuid4().hex
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, rand_hash)
    torch.save(data_dict, ckpt_path)
    return ckpt_path

def save_model_with_args(params, model_info, ckpt_dir):
    """Saves a model with training and hyperparameter info.

    Parameters
    ----------
    params
        Full parameter set for the training script
    model_info
        A dictionary with two entries, "model" and "hyper". "hyper" is the
        parameters for initialization.
    ckpt_dir
        Path to the directory to save this model checkpoint.
    """
    model = model_info['model']
    filtered_args = {}
    for p in inspect.signature(model.__class__.__init__).parameters:
        if p in model_info['hyper']:
            filtered_args[p] = model_info['hyper'][p]
    ckpt_dict = dict(params, **{'state_dict': model.state_dict(),
                                'curr_hyper': filtered_args})
    return save_model(ckpt_dict, ckpt_dir)

def load_model_with_args(cls, ckpt_path):
    ckpt_dict = torch.load(ckpt_path)
    hyper_params = ckpt_dict['curr_hyper']
    model = cls(**hyper_params)
    model.load_state_dict(ckpt_dict['state_dict'])
    return { "model": model, "hyper": hyper_params }

def path_relative_to_main(path):
    """Takes a relative path and computes the absolute path assuming this is
    relative to the directory holding the main script.
    """
    if hasattr(sys.modules['__main__'], '__file__'):
        mainpath = os.path.abspath(sys.modules['__main__'].__file__)
        maindir = os.path.dirname(mainpath)
        return os.path.join(maindir, path)
    else:
        return path

def add_ckpt_config(ckpt_config_path, component, ckpt_path, hyper, train,
                    logpath, parent_ckpt_path=None, addons=None):
    """Adds the given ckeckpoint information to a JSON file.
    This adds the path to the checkpoint, hyperparameters, training arguments,
    and the log path in the following format:
    { "config_tree":
        { <hyper_hash>:
            { <train_hash>:
                { 'hyper': <hyper>,
                  'train': <train>,
                  <timestamp>: {
                    'ckpt_path': <ckpt_path>,
                    'logpath': <logpath>,
                    'component': <component>
                    ...<addons>...
                    'dependents': <ckpt_config> }}}}
      "lookup_table":
        { <ckpt_path>: <list of keys to ckpt location in config info> }}
    The lookup table can be used to find the location of a specific ckeckpoint
    in the configuration tree. The configuration tree groups the configurations
    by hyperparameters and then training parameters, and forms a tree of
    dependents.
    Parameters
    ----------
    ckpt_config_path
        Path to the checkpoint configuration tree file. A new file will be made
        if it does not exist.
    component
        Descriptor for the current compoment (e.g. "type_grammar").
    ckpt_path
        Path to the trained checkpoint being saved.
    hyper
        Dictionary of hyperparameters.
    train
        Dictionary of training parameters.
    logpath
        Path to the log file for this training session.
    parent_ckpt_path
        Path to the parent model (the model that this depends on).
    addons
        Dictionary of additional parameters to save.
    """

    def dict_hash(d):
        return json.dumps(d, sort_keys=True)

    # Make sure all the paths are absolute.
    ckpt_config_path = os.path.abspath(ckpt_config_path)
    ckpt_path = os.path.abspath(ckpt_path)
    logpath = os.path.abspath(logpath)
    if parent_ckpt_path is not None:
        parent_ckpt_path = os.path.abspath(parent_ckpt_path)

    # Lock file so we don't mess if up if we run two scripts at once.
    with FileLock(ckpt_config_path + '.lock'):
        if os.path.isfile(ckpt_config_path):
            config_json = json.loads(open(ckpt_config_path, 'r').read())
        else:
            config_json = { "config_tree": {}, "lookup_table": {} }

        # Get parent.
        if parent_ckpt_path is None:
            parent = None
        elif parent_ckpt_path not in config_json['lookup_table']:
            warnmsg = ("Parent ckeckpoint, {}, not found in the configuration "
                       "lookup table (at {}). Assuming no parent in config.").format(parent_ckpt_path, ckpt_config_path)
            log.warn(warnmsg)
            parent = None
        else:
            parent_keylist = config_json['lookup_table'][parent_ckpt_path]
            parent = config_json['config_tree']
            for key in parent_keylist:
                # Bug in keylist writing code -- skip.
                if key == 'type_grammar':
                    continue
                parent = parent[key]

        # Compute hashes from configs.
        hyper_hash = dict_hash(hyper)
        train_hash = dict_hash(train)

        # Add this to the lookup table.
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        keylist = [hyper_hash, train_hash, timestamp]
        if parent is not None:
            keylist = parent_keylist + ['dependents'] + keylist
        config_json['lookup_table'][ckpt_path] = keylist

        # Add this configuration to the configuration tree.
        root = config_json['config_tree']
        if parent is not None:
            if 'dependents' not in parent:
                parent['dependents'] = {}
            root = parent['dependents']
        if hyper_hash not in root:
            root[hyper_hash] = {}
        if train_hash not in root[hyper_hash]:
            root[hyper_hash][train_hash] = {}
        if 'hyper' not in root[hyper_hash][train_hash]:
            root[hyper_hash][train_hash]['hyper'] = hyper
        if 'train' not in root[hyper_hash][train_hash]:
            root[hyper_hash][train_hash]['train'] = train
        entry = { 'ckpt_path': ckpt_path,
                  'logpath': logpath,
                  'component': component }
        if addons is not None:
            entry.update(addons)
        root[hyper_hash][train_hash][timestamp] = entry

        # Save updated config.
        with open(ckpt_config_path, 'w') as f:
            f.write(json.dumps(config_json, indent=2))

