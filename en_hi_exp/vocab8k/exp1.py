import sys
import os
import shutil
import pathlib
import argparse
import yaml
import getpass

from local_utils import fseq_utils
from local_utils.fseq_utils import fairseq_preprocess, fairseq_train
from local_utils.spm_utils import spm_train, spm_encoder, spm_decoder


def preprocess_data(src_lang, tgt_lang, traindir=None, validdir=None,
                    testdir=None, spm_dir="spm_dir", src_vocab_size=8000, tgt_vocab_size=8000,
                    processed_bin="processed", spm_src_train_data=None, spm_tgt_train_data=None):
    """
    Preprocess data
    """

    data_dirs = []
    if traindir:
        data_dirs.append(("train", traindir))
    if validdir:
        data_dirs.append(("valid", validdir))
    if testdir:
        data_dirs.append(("test", testdir))

    # Check if sentencepiece model dir exists
    base_dir = pathlib.Path(__file__).stem

    def init_spm_models():
        """
        Init sentencepiece models
        """
        if not os.path.exists(spm_dir)
        os.mkdirs(spm_dir)
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

        for data_type, data_dir in data_dirs:
            src_encoded_file_path = os.path.join(
                data_dir,
                f"{data_type}.tok.{src_lang}",
            )
            if not os.path.exists(src_encoded_file_path):
                src_encoded_lines = spm_encoder(
                    spm_src_model_path,
                    os.path.join(data_dir, f"{data_type}.{src_lang}")
                )
                with open(src_encoded_file_path) as wt:
                    wt.write("\n".join(src_encoded_lines))
            tgt_encoded_file_path = os.path.join(
                data_dir,
                f"{data_type}.tok.{tgt_lang}",
            )
            if not os.path.exists(tgt_encoded_file_path):
                tgt_encoded_lines = spm_encoder(
                    spm_tgt_model_path,
                    os.path.join(data_dir, f"{data_type}.{tgt_lang}")
                )
                with open(tgt_encoded_file_path) as wt:
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
        traindir=traindir,
        validdir=validdir,
        testdir=testdir)


def train_model(src_lang, tgt_lang, traindir, validdir, testdir,
                processed_bin="processed", checkpoint_dir="checkpoints", configfile=None,
                local_checkpoints_save=False):
    """
    Train fseq model
    """
    print("Training model")

    base_dir = pathlib.Path(__file__).stem

    if local_checkpoints_save:
        checkpoint_dir = os.path.join(base_dir, checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdirs(checkpoint_dir)
    else:
        username = getpass.getuser()
        ssd_scratch_dir = os.path.join(
            os.sep, "ssd_scratch", "cvit", username)
        if not os.path.exists(ssd_scratch_dir):
            os.mkdir(ssd_scratch_dir)
        checkpoint_dir = os.path.join(
            ssd_scratch_dir,
            os.path.splittext(__file__),
            checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdirs(checkpoint_dir)

    otherargs = {}
    if configfile == None:
        configfile = f"{base_dir}.config.yml"
    otherargs = yaml.load(open(configfile))
    fargs = fseq_utils.FseqArgs(
        os.path.join(base_dir, processed_bin),
        **otherargs,
        save_dir=os.path.join(base_dir, checkpoint_dir),
    )
    fairseq_train(fargs.argslist)


def runall():
    SRC_LANG = "en"
    TGT_LANG = "hi"
    TRAINDIR = ""
    VALIDDIR = ""
    TESTDIR = ""

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
        traindir=TRAINDIR,
        validdir=VALIDDIR,
        testdir=TESTDIR,
    )


if __name__ == "__main__":
    runall()
