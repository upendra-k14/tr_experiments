import sys
import os
import shutil
import pathlib
import argparse
import yaml
import getpass
import subprocess

from local_utils import fseq_utils
from local_utils.fseq_utils import fairseq_preprocess, fairseq_train
from local_utils.spm_utils import spm_train, spm_encoder, spm_decoder


def preprocess_data(src_lang, tgt_lang, traindir=None, validdir=None,
                    testdir=None, spm_dir="spm_dir", src_vocab_size=8000, tgt_vocab_size=8000,
                    processed_bin="processed", spm_src_train_data=None, spm_tgt_train_data=None,
                    encoded_data_dir="tokenized_data"):
    """
    Preprocess data
    """

    # Check if sentencepiece model dir exists
    base_dir = pathlib.Path(__file__).stem

    fseq_train_path = None
    fseq_valid_path = None
    fseq_test_path = None

    data_dirs = []
    if traindir:
        data_dirs.append(("train", traindir, os.path.join(
            base_dir, encoded_data_dir, "train")))
        fseq_train_path = data_dirs[-1][2]
    if validdir:
        data_dirs.append(("valid", validdir, os.path.join(
            base_dir, encoded_data_dir, "valid")))
        fseq_valid_path = data_dirs[-1][2]
    if testdir:
        data_dirs.append(("test", testdir, os.path.join(
            base_dir, encoded_data_dir, "test")))
        fseq_test_path = data_dirs[-1][2]

    def init_spm_models():
        """
        Init sentencepiece models
        """
        if not os.path.exists(spm_dir):
            os.makedirs(spm_dir)
        spm_src_lang_model_path = os.path.join(
            spm_dir, f"{src_lang}.{src_vocab_size}.model")
        spm_tgt_lang_model_path = os.path.join(
            spm_dir, f"{tgt_lang}.{tgt_vocab_size}.model")
        if not os.path.exists(spm_src_lang_model_path):
            spm_src_input = os.path.join(traindir, f"train.{src_lang}")
            if spm_src_train_data:
                spm_src_input = spm_src_train_data
            spm_train(spm_src_input, src_lang, src_vocab_size)
        if not os.path.exists(spm_tgt_lang_model_path):
            spm_tgt_input = os.path.join(traindir, f"train.{tgt_lang}")
            if spm_tgt_train_data:
                spm_tgt_input = spm_tgt_train_data
            spm_train(spm_tgt_input, tgt_lang, tgt_vocab_size)

        return spm_src_lang_model_path, spm_tgt_lang_model_path

    def encode_data(spm_src_model_path, spm_tgt_model_path):
        """
        Encode data using sentencepiece
        """
        enc_data_path = os.path.join(base_dir, encoded_data_dir)
        if not os.path.exists(enc_data_path):
            os.makedirs(enc_data_path)
        for data_type, input_data_dir, output_data_dir in data_dirs:
            if not os.path.exists(output_data_dir):
                os.makedirs(output_data_dir)

        for data_type, in_dir, out_dir in data_dirs:
            src_encoded_file_path = os.path.join(
                out_dir,
                f"{data_type}.tok.{src_lang}",
            )
            if not os.path.exists(src_encoded_file_path):
                src_encoded_tokens = spm_encoder(
                    spm_src_model_path,
                    os.path.join(in_dir, f"{data_type}.{src_lang}")
                )
                src_encoded_lines = (" ".join(x) for x in src_encoded_tokens)
                print(f"Writing tokenized data to {src_encoded_file_path}")
                with open(src_encoded_file_path, "w") as wt:
                    wt.write("\n".join(src_encoded_lines))
            tgt_encoded_file_path = os.path.join(
                out_dir,
                f"{data_type}.tok.{tgt_lang}",
            )
            if not os.path.exists(tgt_encoded_file_path):
                tgt_encoded_tokens = spm_encoder(
                    spm_tgt_model_path,
                    os.path.join(in_dir, f"{data_type}.{tgt_lang}")
                )
                tgt_encoded_lines = (" ".join(x) for x in tgt_encoded_tokens)
                print(f"Writing tokenized data to {tgt_encoded_file_path}")
                with open(tgt_encoded_file_path, "w") as wt:
                    wt.write("\n".join(tgt_encoded_lines))

    print("Init spm models ...")
    spm_src_model_path, spm_tgt_model_path = init_spm_models()
    print("Encoding data ...")
    encode_data(spm_src_model_path, spm_tgt_model_path)
    print("Fairseq preprocessing ...")
    fairseq_preprocess(
        src_lang,
        tgt_lang,
        os.path.join(base_dir, processed_bin),
        traindir=fseq_train_path,
        validdir=fseq_valid_path,
        testdir=fseq_test_path)

def printlog(verbose=True):
    if verbose==True:
        return print
    else:
        def notprint(*args, **kwargs):
            #do nothing
            pass
        return notprint


def train_model(src_lang, tgt_lang, processed_bin="processed", checkpoint_dir="checkpoints",
                configfile=None, local_checkpoints_save=False, last_checkpoint_path=None, 
                printf=printlog(verbose=True)):
    """
    Train fseq model
    """
    print("Training model")

    base_dir = pathlib.Path(__file__).stem
    username = getpass.getuser()
    ssd_scratch_dir = os.path.join(os.sep, "ssd_scratch", "cvit", username)
    home_dir = os.path.join(os.sep, "home", username)
    printf("SSD Scratch dir {}".format(ssd_scratch_dir))
    chkpt_dir = None

    printf("Saving checkpoints locally : {}".format(str(local_checkpoints_save)))
    if local_checkpoints_save == True:
        chkpt_dir = os.path.join(base_dir, checkpoint_dir)
        printf("#Chkpt dir {}".format(chkpt_dir))
        if not os.path.exists(chkpt_dir):
            printf("#Creating dir {}".format(chkpt_dir))
            os.makedirs(chkpt_dir)
    else:
        if not os.path.exists(ssd_scratch_dir):
            os.makedirs(ssd_scratch_dir)
        relative_exp_path = os.path.relpath(
            os.path.splitext(os.path.abspath(__file__))[0],
            home_dir,
        )
        chkpt_dir = os.path.join(ssd_scratch_dir, relative_exp_path, checkpoint_dir)
        printf("##Chkpt dir {}".format(chkpt_dir))
        if not os.path.exists(chkpt_dir):
            printf("##Creating dir {}".format(chkpt_dir))
            os.makedirs(chkpt_dir)

    otherargs = {}
    if configfile == None:
        configfile = f"{base_dir}.config.yml"
    otherargs = yaml.load(open(configfile))
    if last_checkpoint_path != None:
        subprocess.call([
            "rsync",
            "-zavh",
            last_checkpoint_path,
            os.path.join(chkpt_dir, os.sep),
        ])
    print("Checkpoint dir {}".format(chkpt_dir))
    fargs = fseq_utils.FseqArgs(
        os.path.join(base_dir, processed_bin),
        **otherargs,
        save_dir=chkpt_dir,
    )
    fairseq_train(fargs.argslist)


def runall():
    SRC_LANG = "en"
    TGT_LANG = "fr"
    TRAINDIR = "/home/upendra/europarldata/fr_en_data/train"
    VALIDDIR = "/home/upendra/europarldata/fr_en_data/dev"
    TESTDIR = "/home/upendra/europarldata/fr_en_data/test"

    # Preprocess data
    preprocess_data(
        SRC_LANG,
        TGT_LANG,
        traindir=TRAINDIR,
        validdir=VALIDDIR,
        testdir=TESTDIR,
    )

    # Train fseq model
    train_model(
        SRC_LANG,
        TGT_LANG,
        last_checkpoint_path="exp1/checkpoints/checkpoint_last.pt",
    )


if __name__ == "__main__":
    runall()
