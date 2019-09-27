import sys, os, shutil, pathlib, errno
import yaml

from config_exps import *

"""
################################################################################
MANUAL CONFIGS :
################################################################################

#1. Adding mtexp package for importing #########################################
cd mtexp
pip3 install -e . --user
toplevel module : local_utils
################################################################################

#2. Cleaning dir ###############################################################
rm -rf *_exp
################################################################################
"""

def mkdirs(newdir):
    try:
        os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise

def setup_expfiles(experiments):
    """
    Setup experiments and create dirs
    """

    for key, val in experiments.items():
        dirname = os.path.join(*val["exp_name"].split("."))
        if not os.path.exists(dirname):
            mkdirs(dirname)
        else:
            print("{path} already exists", path=dirname)
        initfile = os.path.join(dirname, "__init__.py")
        if not os.path.exists(initfile):
            pathlib.Path(initfile).touch()
        localconfigfile = os.path.join(
            dirname,
            "{}.py".format(LOCAL_CONFIG_FILE))
        if not os.path.exists(localconfigfile):
            localconfigdata = str(LOCAL_CONFIG_DATA)
            with open(localconfigfile, "w") as wt:
                wt.write(localconfigdata)
        descriptionfile = os.path.join(
            dirname, "{expname}{description_suffix}".format(
                expname=val["exp_name"],
                description_suffix=DESCRIPTION_FILE_SUFFIX))
        if not os.path.exists(descriptionfile):
            with open(descriptionfile, "w") as wt:
                yaml.dump(val, wt, default_flow_style=False)
        setupfile = os.path.join(
            dirname, "{expname}_setup.py".format(expname=val["exp_name"]))
        if not os.path.exists(setupfile):
            shutil.copy(__file__, setupfile)
            filedata = open(setupfile).read()
            filedata = filedata.replace(
                "from {} import *\n".format(CONFIG_FILE_NAME),
                "from {} import *\n".format(LOCAL_CONFIG_FILE),
            )
            with open(setupfile, "w") as wt:
                wt.write(filedata)

def main():
    exp_dict = EXPERIMENTS.experiments
    setup_expfiles(exp_dict)

if __name__ == "__main__":
    main()
