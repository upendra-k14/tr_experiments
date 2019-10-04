# Move this file to indic_tagger dir before running
# Use anaconda python3.6 environ with proper packages
# installed using requirements.txt in indic_tagger repo
#
# CRF POS tagger for indic languages.
# Uses pre-trained taggers : https://github.com/avineshpvs/indic_tagger
# Uses modified polyglot_tokenizer : another function tokenize_lines
#
# Example usage :
# python indic_parser.py -l ta -t pos -i <input-data-file> -o <output-data-file> -b 256
from tqdm import tqdm
from polyglot_tokenizer import Tokenizer
from tagger.src import generate_features
from tagger.src.algorithm.CRF import CRF
from functools import partial, lru_cache
import multiprocessing as mp
import threading
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
import lmdb
import time
sys.path.append(path.dirname(path.abspath(__file__)))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{}  {:2.2f} ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed


def profiler(*args, debug=False, **kwargs):
    if debug:
        print(*args, **kwargs)
    else:
        pass
        # Do nothing


def prdebug(*args, debug=False, **kwargs):
    if debug:
        print(*args, **kwargs)
    else:
        pass
        # Do nothing


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
LMDBMULT = 200


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
    parser.add_argument("-i", "--input_file", dest="test_data", type=str, metavar='<str>', required=True,
                        help="Test data path ex: data/test/te/test.txt")
    parser.add_argument("-o", "--output_file", dest="output_path", type=str, metavar='<str>',
                        help="The path to the output file",
                        default=path.join(path.dirname(path.abspath(__file__)), "outputs", "output_file"))
    parser.add_argument("-ot", "--output_type", dest="output_type", type=str, metavar="<str>",
                        help="Type of output : json, pickle (pickle is much faster)",
                        default="pickle")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>',
                        help="Batch size for predicting tags",
                        default=256)
    parser.add_argument("-mp", "--max_processes", dest="max_processes", type=int, metavar='<int>',
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
        textlines = fp.readlines()
        tok = Tokenizer(lang=lang, split_sen=True)
        tokenized_sents = tok.tokenize_lines(textlines)
        for tokens in tokenized_sents:
            sent = []
            for token in tokens:
                # Necessary to use as tuple for caching
                # while generating features based on
                # previous, current and next word
                sent.append((token, "", ""))
            data_tuple.append(sent)
            n_sents += 1
        logger.info("Tokenization done")

    with codecs.open(tokenized_data_path, "wb") as wt:
        logger.info("Writing data into pickle format")
        pickle.dump(data_tuple, wt, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Data written")

    return tokenized_data_path, n_sents


@lru_cache(maxsize=1024)
def index_batch_iterator(batch_size, n_sents):
    """
    Return batch indices
    """

    n_batches = n_sents//batch_size
    if (n_sents % batch_size) != 0:
        n_batches += 1

    low = 0
    high = 0
    range_list = []
    for i in range(n_batches):
        high = low + batch_size
        if high > n_sents:
            high = n_sents
        range_list.append((low, high))
        low = high
    return range_list


def write_anno(filename, X_data, y_data, tag_type, ftype):
    """
    Write annotation to file
    """
    if tag_type == "parse":
        tag_type = "chunk"

    def wopen(fname): return codecs.open(
        fname, "w", encoding="utf8", errors="ignore")
    if ftype == "pickle":
        def wopen(fname): return codecs.open(fname, "wb")

    # Check if lengths are equal
    assert len(X_data) == len(y_data)

    prdebug("length of first sentence ", len(X_data[0]))
    with wopen(filename) as fp:
        data = []
        for i, X_sent in enumerate(X_data):
            jlist = []
            for j, X_token in enumerate(X_sent):
                jlist.append((
                    X_token[0],
                    y_data[i][j] if tag_type == "pos" else X_token[1],
                    y_data[i][j] if tag_type == "chunk" else X_token[2]))
            data.append(jlist)
        prdebug("data length", len(data), len(data[0]))
        if ftype == "pickle":
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            json.dump(data, fp, indent=4)


def print_from(filename):
    """
    Print from file to screen
    """

    with codecs.open(filename, "r", encoding='utf8', errors='ignore') as fp:
        data = json.load(fp)
        for i, x_sent in enumerate(data):
            print(i, json.dumps(x_sent))


def _get_features(db_j, fixed_args):
    try:
        tag_type = fixed_args[0]
        db_path = fixed_args[1]
        db_i = fixed_args[2]
        dbkey = "{}-{}".format(db_i, db_j)
        # Each process will have it's own copy of readonly lmdb env
        # Will be executed only once for each process
        curr_process = mp.current_process()
        if not hasattr(curr_process, "db_object"):
            # print("executed")
            curr_process.db_object = lmdb.open(db_path, readonly=True).begin()
            # curr_process.timer1 = []
            # curr_process.timer2 = []
            # curr_process.timer3 = []
            # if hasattr(curr_process, "db_object"):
            #    print("y")

        db_object = curr_process.db_object
        # Query time is minimal
        binaryobj = db_object.get(dbkey.encode("ascii"))
        sent_item_list = pickle.loads(binaryobj)
        features = [generate_features.sent2features(
            sent_item, tag_type, "crf") for sent_item in sent_item_list]

        #profiler("Curr Avg time for process {} : {:.2f}ms {:.2f}ms {:.2f}ms".format(
        #    curr_process,
        #    np.mean(curr_process.timer1),
        #    np.mean(curr_process.timer3),
        #    np.mean(curr_process.timer2))
        #)
        return features
    except Exception as e:
        traceback.print_exc()
        raise e


@lru_cache(maxsize=1024)
def get_chunks(llen, n_chunks):
    chk_sz = llen//n_chunks
    if llen % n_chunks != 0:
        chk_sz = chk_sz + 1
    l_indices = np.arange(n_chunks)*chk_sz
    h_indices = l_indices + chk_sz
    h_indices[-1] = llen
    results = []
    for low, high in zip(l_indices.tolist(), h_indices.tolist()):
        results.append((low, high))
    return results


def mp_get_features(db_path, db_i, nchunks, tag_type, pool):
    # binding tag_type and model_type from outer scope to _get_features func
    results = pool.imap(
        partial(_get_features, fixed_args=(tag_type, db_path, db_i)),
        range(nchunks))
    X_test = []
    for i, result_chunks in enumerate(results):
        prdebug(i, len(result_chunks), end=" ")
        X_test.extend(result_chunks)
    prdebug()
    return X_test


def batch_predict(args, tag_model_path, chunk_model_path, wt_screen=False):
    """
    Predict in batches
    Optimized for predicting POS tags in batches and uses multiprocessing
    """
    mp_n_procs = min(args.max_processes, N_PROCESSES)
    logger.info("N PROCESSES: {}".format(mp_n_procs))
    pool = mp.Pool(processes=mp_n_procs)

    batch_size = args.batch_size
    logger.info("tokenizing all data at once")
    tk_data_path, n_sents = tokenize_data(args.test_data, args.language)
    logger.info("N SENTENCES: {}".format(n_sents))
    n_batches = n_sents//batch_size
    if (n_sents % batch_size) != 0:
        n_batches += 1

    # Code for optimizing I/O in multiprocessing
    ############################################################################
    map_size = path.getsize(args.test_data)*LMDBMULT
    if path.exists("lmdbcache/"):
        from shutil import rmtree
        rmtree("lmdbcache/")
    env = lmdb.open("lmdbcache", map_size=map_size)
    logger.info("Caching data using lmdb with map size {:.2f} MB".format(
        map_size/(1024.0*1024.0)))
    with env.begin(write=True) as txn:
        data_tuples = []
        with codecs.open(tk_data_path, "rb") as fp:
            data_tuples = pickle.load(fp)
        ts = time.time()
        for i, (low, high) in enumerate(index_batch_iterator(batch_size, n_sents)):
            tempi = data_tuples[low:high]
            for j, (ch_low, ch_high) in enumerate(get_chunks(high-low, mp_n_procs)):
                tempij = pickle.dumps(
                    tempi[ch_low:ch_high], protocol=pickle.HIGHEST_PROTOCOL)
                lmdbkey = "{}-{}".format(i, j)
                txn.put(lmdbkey.encode("ascii"), tempij)
        te = time.time()
        logger.info("Total time for caching {:.2f} ms".format((te-ts)*1000.0))
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
    db_path = "lmdbcache"
    with tqdm(total=n_batches) as progress_bar:
        for ni, (nlow, nhigh) in enumerate(index_batch_iterator(batch_size, n_sents)):
            y_pred = None
            if args.tag_type == "parse":
                X_test = mp_get_features(db_path, ni, mp_n_procs, "pos", pool)
                y_pos = tagger.predict(X_test)
                test_sents_pos = generate_features.append_tags(
                    test_sents, "pos", y_pos)
                X_test = mp_get_features(
                    db_path, ni, mp_n_procs, "chunk", pool)
                y_pred = chunker.predict(X_test)
            else:
                X_test = mp_get_features(
                    db_path, ni, mp_n_procs, args.tag_type, pool)
                y_pred = crf_tagger.predict(X_test)
            progress_bar.update(1)

            # Append batch results
            all_y_pred.extend(y_pred)

    # Check if all sentences are processed
    assert n_sents == len(all_y_pred)

    logger.info("Writing results to {}".format(args.output_path))
    write_anno(args.output_path, data_tuples, all_y_pred,
               args.tag_type, args.output_type)
    logger.info("Writing done")

    if wt_screen:
        print_from(args.output_path)
    logger.info("Output in: {}".format(args.output_path))


def main():
    """
    CRF Tagger : POS, Chunk and Parse
    """

    # curr_dir = path.dirname(path.abspath(__file__))
    curr_dir = os.getcwd()
    args = get_args()

    tag_model_path = "{}/models/{}/{}.{}.{}.model".format(
        curr_dir, args.language, "crf", "pos", "utf")
    chunk_model_path = "{}/models/{}/{}.{}.{}.model".format(
        curr_dir, args.language, "crf", "chunk", "utf")

    batch_predict(args, tag_model_path, chunk_model_path)


if __name__ == "__main__":
    main()
