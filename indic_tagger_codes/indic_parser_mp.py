# Move this file to indic_tagger dir before running
# Use anaconda python3.6 environ with proper packages
# installed using requirements.txt in indic_tagger repo
#
# CRF POS tagger for indic languages.
# Uses pre-trained taggers : https://github.com/avineshpvs/indic_tagger
from tqdm import tqdm
from polyglot_tokenizer import Tokenizer
from tagger.src import generate_features
from tagger.src.algorithm.CRF import CRF
from functools import partial
import multiprocessing as mp
import json
import pickle
import codecs
import logging
import argparse
import sys
import os.path as path
import os
import re
import numpy as np
import traceback
sys.path.append(path.dirname(path.abspath(__file__)))


LOGFORMAT = "{asctime}: {levelname}: {funcName:15s}: {message}"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
loghandler = logging.StreamHandler()
lf = logging.Formatter(LOGFORMAT, style="{")
loghandler.setFormatter(lf)
logger.addHandler(loghandler)

# number of logical cpus (hyperthreaded)
# N_CPU = mp.cpu_count()
# number of physical cpus
N_CPU = len(os.sched_getaffinity(0))
N_PROCESSES = max(1, N_CPU - 1)


def get_args():
    """
    This function parses and return arguments passed in
    Uses code from : https://github.com/avineshpvs/indic_tagger
    """

    parser = argparse.ArgumentParser(
        description='Indic lang tagger/chunker/parser')
    parser.add_argument("-l", "--language", dest="language", type=str, metavar='<str>', required=True,
                        help="Language of the dataset: te (telugu), hi (hindi), ta (tamil), ka (kannada), pu (punjabi), mr (Marathi), be (Bengali), ur (Urdu), ml (Malayalam)")
    parser.add_argument("-t", "--tag_type", dest="tag_type", type=str, metavar='<str>', required=True,
                        help="Tag type: pos, chunk, parse")
    parser.add_argument("-i", "--input_file", dest="test_data", type=str, metavar='<str>', required=False,
                        help="Test data path ex: data/test/te/test.txt")
    parser.add_argument("-o", "--output_file", dest="output_path", type=str, metavar='<str>',
                        help="The path to the output file",
                        default=path.join(path.dirname(path.abspath(__file__)), "outputs", "output_file"))
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>',
                        help="Batch size for predicting tags",
                        default=128)
    parser.add_argument("-m", "--max_processes", dest="max_processes", type=int, metavar='<int>',
                        help="Max number of processes",
                        default=N_PROCESSES)
    return parser.parse_args()


def tokenize_data(data_path, lang, forcesave=False):
    """
    Load and save tokenized data
    """

    tokenized_data_path = data_path+".ptok"
    if not forcesave:
        if path.exists(tokenized_data_path):
            logger.info("Tokenized file already present at {}".format(
                tokenized_data_path))
            n_sents = 0
            with codecs.open(tokenized_data_path, "rb") as fp:
                n_sents = len(pickle.load(fp))
            return tokenized_data_path, n_sents

    data_tuple = []
    n_sents = 0
    with codecs.open(data_path, 'r', encoding='utf-8') as fp:
        logger.info("Loading whole data in memory ...")
        text = fp.read()
        tok = Tokenizer(lang=lang, split_sen=True)
        tokenized_sents = tok.tokenize(text)
        sent = []
        for tokens in tokenized_sents:
            for token in tokens:
                sent.append([token, "", ""])
            data_tuple.append(sent)
            n_sents += 1
        logger.info("Tokenization done")

    with codecs.open(tokenized_data_path, "wb") as wt:
        logger.info("Writing data into pickle format")
        # dataline_str = "\n".join([json.dumps(x) for x in data_tuple])
        # wt.write(dataline_str)
        pickle.dump(data_tuple, wt, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Data written")

    return tokenized_data_path, n_sents


def batch_iterator(data_path, batch_size, n_sents):
    """
    Generator for getting batched data
    """

    n_batches = n_sents//batch_size
    if (n_sents % batch_size) != 0:
        n_batches += 1
    data_tuples = []
    # logger.info("Loading data")
    with codecs.open(data_path, "rb") as fp:
        data_tuples = pickle.load(fp)
    low = 0
    high = 0
    for i in range(n_batches):
        high = low + batch_size
        if high > n_sents:
            high = n_sents
        yield (data_tuples[low:high], high - low)
        low = high


def write_anno(filename, X_data, y_data, tag_type):
    """
    Modified code from :
    https://github.com/avineshpvs/indic_tagger/blob/master/tagger/utils/writer.py
    """
    if tag_type == "parse":
        tag_type = "chunk"

    with codecs.open(filename, "w", encoding='utf8', errors='ignore') as fp:
        data = []
        for i, X_sent in enumerate(X_data):
            jlist = []
            for j, X_token in enumerate(X_sent):
                if tag_type == "pos":
                    X_token[1] = y_data[i][j]
                if tag_type == "chunk":
                    X_token[2] = y_data[i][j]
                jlist.append(X_token)
            data.append(jlist)
        json.dump(data, fp)


def print_from(filename):
    """
    Print from file to screen
    """

    with codecs.open(filename, "r", encoding='utf8', errors='ignore') as fp:
        data = json.load(fp)
        for i, x_sent in enumerate(data):
            print(i, json.dumps(x_sent))


class LE():

    def __init__(self, data):
        self.data = data


def _get_features(sent_item_list, tag_type):
    try:
        features = [generate_features.sent2features(
            sent_item, tag_type, "crf") for sent_item in sent_item_list.data]
        return features
    except Exception as e:
        traceback.print_exc()
        raise e


def batch_predict(args, tag_model_path, chunk_model_path, wt_screen=False):
    """
    Predict in batches
    Optimized for predicting POS tags in batches and uses multiprocessing
    """

    batch_size = args.batch_size
    logger.info("tokenizing all data at once")
    tk_data_path, n_sents = tokenize_data(args.test_data, args.language)
    logger.info("N SENTENCES: {}".format(n_sents))
    n_batches = n_sents//batch_size
    if (n_sents % batch_size) != 0:
        n_batches += 1

    # Helper funcs for multiprocessing
    ############################################################################
    mp_n_procs = min(args.max_processes, N_PROCESSES)
    logger.info("N PROCESSES: {}".format(mp_n_procs))

    def get_chunks(sentlist, llen, n_chunks):
        chk_sz = llen//n_chunks
        l_indices = np.arange(n_chunks)*chk_sz
        h_indices = l_indices + chk_sz
        h_indices[-1] = -1
        for low, high in zip(l_indices.tolist(), h_indices.tolist()):
            yield sentlist[low:high]

    def mp_get_features(test_sents, bsz, tag_type):
        pool = mp.Pool(processes=mp_n_procs)
        mp_chunk_sz = max(2, bsz//(mp_n_procs + 1))
        # binding tag_type and model_type from outer scope to _get_features func
        results = pool.imap(
            partial(_get_features, tag_type=tag_type),
            [LE(x) for x in get_chunks(test_sents, bsz, mp_n_procs)],
        )
        X_test = []
        for result_chunks in results:
            X_test.extend(result_chunks)
        return X_test
    ############################################################################

    all_test_sents = []
    all_y_pred = []

    # POS tagger
    tagger = None
    if args.tag_type != "chunk":
        tagger = CRF(tag_model_path)
        tagger.load_model()
        logger.info("Loaded CRF model from {}".format(tag_model_path))

    # Chunker
    chunker = None
    if args.tag_type == "parse" or args.tag_type == "chunk":
        chunker = CRF(chunk_model_path)
        chunker.load_model()
        logger.info("Loaded CRF model from {}".format(chunk_model_path))

    crf_tagger = None
    if args.tag_type == "pos":
        crf_tagger = tagger
    elif args.tag_type == "chunk":
        crf_tagger = chunker

    logger.info("Generating features and predicting in batches")
    logger.info("Total batches : {}, Batch size : {}".format(
        n_batches, batch_size))
    with tqdm(total=n_batches) as progress_bar:
        for i, (test_sents, bsz) in enumerate(batch_iterator(tk_data_path, batch_size, n_sents)):
            y_pred = None
            if args.tag_type == "parse":
                X_test = mp_get_features(test_sents, bsz, "pos")
                y_pos = tagger.predict(X_test)
                test_sents_pos = generate_features.append_tags(
                    test_sents, "pos", y_pos)
                X_test = mp_get_features(test_sents_pos, bsz, "chunk")
                y_pred = chunker.predict(X_test)
            else:
                # print("Batch f {}".format(i))
                X_test = mp_get_features(test_sents, bsz, args.tag_type)
                # print("Batch p {}".format(i))
                y_pred = crf_tagger.predict(X_test)
            progress_bar.update(1)

            # Append batch results
            all_test_sents.extend(test_sents)
            all_y_pred.extend(y_pred)

    # Check if all sentences are processed
    assert n_sents == len(all_y_pred)

    logger.info("Writing results to {}".format(args.output_path))
    write_anno(args.output_path, all_test_sents, all_y_pred, args.tag_type)
    logger.info("Writing done")

    if wt_screen:
        print_from(args.output_path)
    logger.info("Output in: {}".format(args.output_path))


def main():
    """
    CRF Tagger : POS, Chunk and Parse
    """

    curr_dir = path.dirname(path.abspath(__file__))
    args = get_args()

    output_dir = path.join(path.dirname(path.abspath(__file__)), "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tag_model_path = "{}/models/{}/{}.{}.{}.model".format(
        curr_dir, args.language, "crf", "pos", "utf")
    chunk_model_path = "{}/models/{}/{}.{}.{}.model".format(
        curr_dir, args.language, "crf", "chunk", "utf")

    batch_predict(args, tag_model_path, chunk_model_path)


if __name__ == "__main__":
    main()
