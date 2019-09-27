from local_utils.experiments import Experiments
from pathlib import Path

LOCAL_CONFIG_FILE = "localconfig"

def get_template():
    template_lines = open(__file__).readlines()
    data = "".join(template_lines[:19])
    return data

LOCAL_CONFIG_DATA = get_template()

DESCRIPTION_FILE_SUFFIX = "_exp.description.yml"
CONFIG_FILE_NAME = Path(__file__).stem
EXPERIMENTS = Experiments()

###############################################################################
# MODIFY BELOW
###############################################################################

task1_args = {
    "description" : "English to Hindi Translation Task",
    "src_lang" : "en",
    "tgt_lang" : "hi",
    "exp_name" : "en_hi_exp",
}
EXPERIMENTS.add_new_task("en_hi", **task1_args)

task2_args = {
    "description" : "Hindi to English Translation Task",
    "src_lang" : "hi",
    "tgt_lang" : "en",
    "exp_name" : "hi_en_exp",
}
EXPERIMENTS.add_new_task("hi_en", **task2_args)

"""
task3_args = {
    "description" : "English to Tamil Translation Task",
    "src_lang" : "en",
    "tgt_lang" : "ta",
    "exp_name" : "en_ta_exp",
}
EXPERIMENTS.add_new_task("en_ta", **task3_args)

task4_args = {
    "description" : "Tamil to English Translation Task",
    "src_lang" : "ta",
    "tgt_lang" : "en",
    "exp_name" : "ta_en_exp",
}
EXPERIMENTS.add_new_task("ta_en", **task4_args)
"""

__all__ = [
    'LOCAL_CONFIG_FILE',
    'LOCAL_CONFIG_DATA',
    'DESCRIPTION_FILE_SUFFIX',
    'CONFIG_FILE_NAME',
    'EXPERIMENTS',
]
