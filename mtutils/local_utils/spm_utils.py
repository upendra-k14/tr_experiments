import os

import sentencepiece as spm
from tqdm import tqdm

def spm_train(spm_input, spm_lang, spm_vocab_size, spm_path="spm_dir", spm_char_coverage=1.0):
    """
    Trainer for sentencepiece model
    """
    spm.SentencePieceTrainer.Train(
        f"--input={spm_input} \
        --model_prefix={spm_path}{os.path.sep}{spm_lang}.{spm_vocab_size} \
        --vocab_size={spm_vocab_size} \
        --character_coverage={spm_char_coverage}"
    )

def spm_encoder(spm_model_path, data_file, spm_output_format="piece"):
    """
    Encoder for sentencepiece
    """
    sp = spm.SentencePieceProcessor()
    print(f"Loading spm model from {spm_model_path}")
    sp.Load(spm_model_path)
    lines = open(data_file).readlines()

    encoder = sp.EncodeAsPieces
    if spm_output_format == "id":
        encoder = sp.EncodeAsIds

    output_lines = []
    #print("Encoding ... ")
    for line_data in lines:
        # Remove \n character before encoding
        encoded_output = encoder(line_data[:-1])
        yield encoded_output

def spm_decoder(spm_model_path, data_file, spm_input_format="piece"):
    """
    Decoder for sentencepiece
    """

    sp = spm.SentencePieceProcessor()
    print(f"Loading spm model from {spm_model_path}")
    sp.Load(spm_model_path)
    lines = open(data_file).readlines()

    decoder = sp.DecodePieces
    if spm_input_format == "id":
        decoder = sp.DecodeIds

    output_lines = []
    #print("Decoding ...")
    for line_data in lines:
        # Remove \n character before decoder
        decoded_output = decoder(line_data[:-1])
        yield decoded_output
