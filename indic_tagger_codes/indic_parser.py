# Move this file to indic_tagger dir before running
# Use anaconda python3.6 environ with proper packages
# installed using requirements.txt in indic_tagger repo
#
# CRF POS tagger for indic languages.
# Uses pre-trained taggers : https://github.com/avineshpvs/indic_tagger
import sys, os.path as path
import os, re
sys.path.append(path.dirname(path.abspath(__file__)))
import argparse
import logging
import codecs
import pickle
import json
import multiprocessing as mp
from functools import partial

# Tagger modules
from tagger.src.algorithm.CRF import CRF
from tagger.src import generate_features
from polyglot_tokenizer import Tokenizer

import tqdm
logger = logging.getLogger(__name__)

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

    parser = argparse.ArgumentParser(description='Indic lang tagger/chunker/parser')
    parser.add_argument("-l", "--language", dest="language", type=str, metavar='<str>', required=True,
                         help="Language of the dataset: te (telugu), hi (hindi), ta (tamil), ka (kannada), pu (pubjabi), mr (Marathi), be (Bengali), ur (Urdu), ml (Malayalam)")
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

def tokenize_data(data_path, lang):
    """
    Load and save tokenized data
    """
    data_tuple = []
    n_sents = 0
    with codecs.open(filename, 'r', encoding='utf-8') as fp:
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

    tokenized_data_path = filename+".tok"
    with codecs.open(tokenized_data_path, "w", encoding="utf-8") as wt:
        logger.info("Writing data into json format line by line")
        dataline_str = "\n".join([json.dumps(x) for x in data_tuple])
        wt.write(dataline_str)

    return tokenized_data_path, n_sents

def batch_iterator(data_path, batch_size, n_sents):
    """
    Generator for getting batched data
    """

    with codecs.open(data_path, "r", encoding="utf-8") as fp:
        total_lines_read = 0
        batch_counter = 0
        j = 0
        sentence_list = None
        sentence_list_len = 0
        num_sents_to_read = 0
        for fpline in fp:
            num_sents_left = n_sents - total_lines_read
            # Start of batch
            if(j==0):
                num_sents_to_read = batch_size
                if num_sents_left < batch_size:
                    num_sents_to_read = num_sents_left
                sentence_list_len = num_sents_to_read
                sentence_list = [0]*sentence_list_len

            # Remove \n character : fpline[:-1]
            sentence_list[j] = json.loads(fpline[:-1])
            # increment number of lines read
            total_lines_read = total_lines_read + 1
            # decrement number of lines to read in current batch
            num_sents_to_read = num_sents_to_read - 1

            # End of batch
            if(j==(num_sents_to_read-1)):
                j = 0
                batch_counter = batch_counter + 1
                yield sentence_list, sentence_list_len

            # End of loop condition
            num_sents_left = n_sents - total_lines_read
            if num_sents_left <=0 :
                break

def write_anno(filename, X_data, y_data, tag_type):
    """
    Modified code from :
    https://github.com/avineshpvs/indic_tagger/blob/master/tagger/utils/writer.py
    """

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

def batch_predict(args, tag_model_path, chunk_model_path, wt_screen=False):
    """
    Predict in batches
    Optimized for predicting POS tags in batches and uses multiprocessing
    """

    batch_size = args.batch_size
    tk_data_path, n_sents = tokenize_data(args.test_data, args.language)
    n_batches = n_sents//batch_size
    if (n_sents % batch_size) != 0:
        n_batches += 1

    # Helper funcs for multiprocessing
    ############################################################################
    mp_n_procs = min(args.max_processes, N_PROCESSES)
    _get_features = lambda tag_type, model_type, sent_list : [
        generate_features.sent2features(
            s, tag_type, model_type) for s in test_sents
    ]
    mp_list2batches = lambda mplist, mp_nprocesses, mplist_bsz : map(
        lambda low, high : mplist[low:high],
        map(
            lambda st, end : (st, end) if end<=mplist_len else (st,mplist_len),
            map(
                lambda l : (l,l+mplist_bsz),
                map(lambda i: i*mplist_bsz, range(0, mp_nprocesses))
            )
        )
    )
    def mp_get_features(test_sents, bsz, tag_type):
        pool = mp.Pool(processes=mp_n_procs)
        mp_bsz = 1 + bsz//mp_n_procs if (bsz%mp_n_procs!=0) else bsz//mp_n_procs
        # binding tag_type and model_type from outer scope to _get_features func
        X_test = pool.map(
            partial(_get_features, tag_type, args.model_type),
            mp_list2batches(test_sents, mp_n_procs, mp_bsz))
        pool.close()
        pool.join()
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
    else:
        args.tag_type == "chunk":
        crf_tagger = chunker

    logger.info("Generating features and predicting in batches")
    logger.info("Total batches : {}, Batch size : {}".format(
        n_batches, batch_size))
    with tqdm(total=n_batches, mininterval=20.0) as progress_bar:
        for test_sents, bsz in batch_iterator(tk_data_path, batch_size, n_sents):
            y_pred = None
            if args.tag_type == "parse":
                X_test = mp_get_features(test_sents, bsz, "pos")
                y_pos = tagger.predict(X_test)
                test_sents_pos = generate_features.append_tags(
                    test_sents, "pos", y_pos)
                X_test = mp_get_features(test_sents_pos, bsz, "chunk")
            else:
                X_test = mp_get_features(test_sents, bsz, args.tag_type)
                y_pred = crf_tagger.predict(X_test)
            progress_bar.update(1)

            # Append batch results
            all_test_sents.extend(test_sents)
            all_y_pred.extend(y_pred)

    # Check if all sentences are processed
    assert n_sents == len(all_y_pred)

    logger.info("Writing: Results to {}".format(args.output_path))
    write_anno(args.output_path, all_test_sents, all_y_pred, args.tag_type)
    logger.info("Writing: Done")

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
        curr_dir, args.language, "crf", "pos", "utf8")
    chunk_model_path = "{}/models/{}/{}.{}.{}.model".format(
        curr_dir, args.language, "crf", "chunk", "utf8")

    batch_predict(args, tag_model_path, chunk_model_path)

if __name__ == "__main__":
    main()
