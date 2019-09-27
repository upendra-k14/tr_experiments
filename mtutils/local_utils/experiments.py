from copy import deepcopy

class Experiments(object):

    def __init__(self):
        self.experiments = dict()

    def query_nestedpath(self, taskname):
        taskpath = taskname.split(".")
        tempdict = deepcopy(self.experiments)
        for _key in taskpath[:-1]:
            if _key in tempdict:
                tempdict = tempdict[_key]
            else:
                return False
        lastkey = taskpath[-1]
        if lastkey in tempdict:
            return True
        else:
            return False

    def add_new_task(self, taskname, **kwargs):
        taskpath = taskname.split(".")
        tempdict = deepcopy(self.experiments)
        for _key in taskpath[:-1]:
            if _key in tempdict:
                tempdict = tempdict[_key]
            else:
                tempdict = dict()
        lastkey = taskpath[-1]
        if lastkey in tempdict:
            print("Duplicate taskname")
        else:
            tempdict[lastkey] = kwargs
        self.experiments = tempdict
