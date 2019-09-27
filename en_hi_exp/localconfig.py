from local_utils.experiments import Experiments
from pathlib import Path

LOCAL_CONFIG_FILE = "localconfig"

def get_template():
    template_lines = open(__file__).readlines()
    data = "".join(template_lines[:20])
    return data

LOCAL_CONFIG_DATA = get_template()

DESCRIPTION_FILE_SUFFIX = "_exp.description.yml"
CONFIG_FILE_NAME = Path(__file__).stem

###############################################################################
# MODIFY BELOW
###############################################################################
EXPERIMENTS = Experiments()

task1_args = {
    "description" : "English to Hindi Translation Task (vocab8k)",
    "src_lang" : "en",
    "tgt_lang" : "hi",
    "exp_name" : "vocab8k",
}
EXPERIMENTS.add_new_task("en_hi.vocab8k", **task1_args)
