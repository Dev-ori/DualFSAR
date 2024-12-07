import yaml

dataset_registry = {}

def dataset_register(name=None):
    def decorator(obj):
        dataset_registry[name or obj.__name__] = obj
        return obj
    return decorator

def build_dataset(cfg, mode, batch_size, global_rank, world_size):
    with open(cfg, mode = 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if batch_size is not None:
        cfg['BATCH_SIZE'] = batch_size
        
    cfg = cfg[mode.upper()]
    dataset = cfg['DATASET']

    get_dataset = dataset_registry.get(dataset.lower(), None)
    if get_dataset is None:
        raise Exception(f"'dataset' doesn't exist. dataset's input must be {', '.join(dataset_registry.keys())}")

    return get_dataset(cfg, mode, global_rank, world_size), cfg