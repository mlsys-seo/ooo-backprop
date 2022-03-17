import argparse
import os

_GLOBAL_ARGS = None

def get_args():
    """Return arguments."""
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        set_args()
    return _GLOBAL_ARGS

def set_args():
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args()

def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='OOO backprop Arguments')

    # Standard arguments.
    parser = _add_OOO_args(parser)
    parser = _add_training_args(parser)
    parser = _add_debugging_args(parser)

    args, unknown_args = parser.parse_known_args()

    _print_args(args, unknown_args)
    
    return args

def _add_OOO_args(parser):
    group = parser.add_argument_group(title='OOO args')
    group.add_argument('--reverse_first_k', type=int, default=0,
                       help='reverse_first_k.')
    group.add_argument('--async_op', type=str, default="_LATER",
                       help='async_op.')
    group.add_argument('--prefix', type=str, default="LAYER_",
                       help='')
    group.add_argument('--separater', type=str, default="_END",
                       help='separater')
    group.add_argument('--num_dependency', type=int, default=3,
                       help='num_dependency')
    return parser

def _add_training_args(parser):
    group = parser.add_argument_group(title='training args')
    group.add_argument('--model_size', type=int, default=50,
                       help='batch_size.')
    group.add_argument('--batch_size', type=int, default=32,
                       help='batch_size.')
    group.add_argument('--num_training_step', type=int, default=30,
                       help='batch_size.')
    return parser

def _add_debugging_args(parser):
    group = parser.add_argument_group(title='training args')
    group.add_argument('--debug_print', type=bool, default=False,
                       help='whether print debug message.')
    return parser

def _print_args(args, unknown_args):
    """Print arguments."""
    print('\n\n------------------------ arguments ------------------------',
            flush=True)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print('-------------------- end of arguments ---------------------\n\n',
            flush=True)

    if len(unknown_args):
        print('\n\n-------------------- unknown arguments --------------------', flush=True)
        for idx in range(0, len(unknown_args), 2):
            dots = '.' * (48 - len(unknown_args[idx]))
            print('  {} {} {}'.format(unknown_args[idx], dots, unknown_args[idx+1]))
        print('----------------- end of unknown arguments ----------------\n\n', flush=True)
