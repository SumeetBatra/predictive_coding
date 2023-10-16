import wandb


def config_wandb(**kwargs):
    # wandb initialization
    wandb.init(project=kwargs['wandb_project'], entity=kwargs['entity'], \
               group=kwargs['wandb_group'], name=kwargs['run_name'], \
               tags=[kwargs['tags']])
    cfg = kwargs.get('cfg', None)
    if cfg is None:
        cfg = {}
        for key, val in kwargs.items():
            cfg[key] = val
    wandb.config.update(cfg)