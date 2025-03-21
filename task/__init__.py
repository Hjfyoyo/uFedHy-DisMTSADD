from torch import nn

# from algorithms.DeepSVDD.SVDD import Model_first_stage, Model_second_stage
from options import args
# if args.tsadalg == 'deep_svdd':
#     from task.SVDD import config_svdd

if args.dataset == 'smd':
    from task.smd_MOON import *
elif args.dataset == 'smap':
    from task.smap_MOON import *
elif args.dataset == 'psm':
    from task.psm_MOON import *
elif args.dataset == 'swat':
    from task.swat_MOON import *
elif args.dataset == 'wadi':
    from task.wadi_MOON import *
elif args.dataset == 'skab':
    from task.skab_MOON import *
elif args.dataset == 'msl':
    from task.msl_MOON import *
elif args.dataset == 'msds':
    from task.msds_MOON import *

def load_model(state_dict,strict=True) -> nn.Module:
    model = model_fun()
    model.load_state_dict(state_dict, strict=strict)
    return model


logger.log_config(config)
print(args)
print(config)
# logger.print(f"local_optimizer:\n{config['optimizer_fun'](model_fun().parameters())} ")
