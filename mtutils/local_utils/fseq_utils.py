import os, pathlib
import math
import random

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.models import (
    transformer,
    register_model,
    register_model_architecture,
)
from fairseq_cli import preprocess, train

class FseqArgs():
    """
    Args class for avoiding use of command-line arguments
    """

    def __init__(self, *args, **kwargs):
        """
        Create args and kwargs list
        """

        self.args_list = []
        self.args_list.extend(args)

        for key, val in kwargs.items():
            newkey = key.replace("_","-")
            if isinstance(val, bool):
                self.args_list.append(f"--{newkey}")
            else:
                self.args_list.append(f"--{newkey}={val}")

    @property
    def argslist(self):
        return self.args_list

def fairseq_preprocess(src_lang, tgt_lang, destdir, traindir=None,
    validdir=None, testdir=None):
    """
    Helper function to do pre-processing using fairseq-preprocess
    """

    def preprocessing_done():
        if os.path.exists(destdir):
            # TODO : more extensive checks
            print("Warning: Check processed dir manually")
            return True
        else:
            return False

    if not preprocessing_done():
        # TODO : to use FseqArgs
        args = []
        args.append(f"--source-lang={src_lang}")
        #src_dict_path = os.path.join(destdir, f"dict.{src_lang}.txt")
        #args.append(f"--srcdict={src_dict_path}")
        args.append(f"--target-lang={tgt_lang}")
        #tgt_dict_path = os.path.join(destdir, f"dict.{tgt_lang}.txt")
        #args.append(f"--tgtdict={tgt_dict_path}")
        if traindir:
            args.append(f"--trainpref={traindir}/train.tok")
        if validdir:
            args.append(f"--validpref={validdir}/valid.tok")
        if testdir:
            args.append(f"--testpref={testdir}/test.tok")
        args.append(f"--destdir={destdir}")

        # fairseq preprocessing argument parser
        parser = options.get_preprocessing_parser()
        pargs = parser.parse_args(args)
        preprocess.main(pargs)
    else:
        print("Probably, preprocessing is already done. Check dirs.")

def fairseq_train(input_args):
    """
    Helper function for training
    """

    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, input_args=input_args)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=train.distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            train.distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=train.distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        train.main(args)
