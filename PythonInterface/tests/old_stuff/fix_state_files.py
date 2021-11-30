import json
import math
from pathlib import Path

import os
import ntpath
import re

from dplus.Amplitudes import amp_to_ampj_converter


class TreeTraversal:
    def fixer_func(self, tree):
        raise NotImplemented
    def traverse_tree(self, tree):
        self.fixer_func(tree)
        if isinstance(tree, dict):
            for key in tree:
                try:
                    if isinstance(tree[key], dict):
                        self.traverse_tree(tree[key])
                    if isinstance(tree[key], list):
                        for item in tree[key]:
                            self.traverse_tree(item)
                except TypeError:
                    pass
                except Exception as e:
                    print(e)
        if isinstance(tree, list):
            return


class fix_path_names(TreeTraversal):
    '''
    This function fixes path names ON THE ASSUMPTION that all relevant path names are contained in the same folder as
    the state file that is being fixed
    '''
    def __init__(self, base_path):
        self.base_path=base_path

    def fixer_func(self, tree):
        try:
            if isinstance(tree, dict):
                for key in tree:
                    if isinstance(tree[key], str):
                        if "\\" in tree[key] or "/" in  tree[key]:  #\\ for windows ,/ for linux path view
                            base_path = os.path.dirname(tree[key])
                            base_name=ntpath.basename(tree[key])
                            new_path=os.path.join(self.base_path, base_name)
                            tree[key]=new_path
            if isinstance(tree, list):
                for i in range(len(tree)):
                    if isinstance(tree[i], str):
                        if "\\" in tree[i] or "/" in  tree[key]:  #\\ for windows ,/ for linux path view
                            base_path = os.path.dirname(tree[i])
                            base_name = ntpath.basename(tree[i])
                            new_path = os.path.join(self.base_path, base_name)
                            tree[i] = new_path

        except TypeError:
            pass


class fix_missing_fields(TreeTraversal):
    '''
    Some state files from before parameter validation was added have a missing TrustRegionStrategyType.
    This fixes that by arbitrarily assigning the type to be Levenberg-Marquardt.
    Obviously if the user wanted a non-arbitrary type, they shouldn't have left it blank.
    '''
    def fixer_func(self, tree):
        try:
            if isinstance(tree, list):
                return
            if tree["MinimizerType"]=="Trust Region":
                if tree["TrustRegionStrategyType"]=="":
                    tree["TrustRegionStrategyType"]="Levenberg-Marquardt"
        except KeyError:
            pass

class fix_bad_limits(TreeTraversal):
    '''
    Before limit validation was added, some paramters had their min and max default limits set to 0.
    this fixes that by replacing them with the new defaults of infinity (but not when only one is zero)
    '''
    def fixer_func(self, tree):
        try:
            if isinstance(tree, list):
                return
            if tree["MaxValue"]==0 and tree["MinValue"]==0:
                tree["MaxValue"]=math.inf
                tree["MinValue"]=-math.inf
        except KeyError:
            pass

class fix_ampj(TreeTraversal):
    def add_j(self, tree, key):
        ext = Path(tree[key]).suffix
        if ext == ".amp":
            ampj_path=tree[key] + "j"
            if True: #not os.path.exists(ampj_path):
                ampj_path = amp_to_ampj_converter(tree[key])
            tree[key] = ampj_path
    def fixer_func(self, tree):
            try:
                if isinstance(tree, dict):
                    for key in tree:
                        if isinstance(tree[key], str):
                            if "\\" in tree[key] or "/" in tree[key]: #\\ for windows ,/ for linux path view
                                self.add_j(tree, key)
                if isinstance(tree, list):
                    for i in range(len(tree)):
                        if isinstance(tree[i], str):
                            if "\\" in tree[i] or "/" in tree[key]:  #\\ for windows ,/ for linux path view
                                self.add_j(tree, i)

            except TypeError:
                pass

def fix_state(state, base_path):
    p=fix_path_names(base_path)
    f=fix_missing_fields()
    l=fix_bad_limits()
    j=fix_ampj()
    p.traverse_tree(state)
    f.traverse_tree(state)
    l.traverse_tree(state)
    j.traverse_tree(state)
    return state

def fix_file(filename):
    with open(filename, 'r') as file:
        my_str=file.read()

    state=json.loads(my_str)
    base_path=os.path.dirname(filename)
    state=fix_state(state, base_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    new_name= os.path.join(base_path, base_name+"_fixed.state")
    with open(new_name, 'w') as file:
        json.dump(state, file)
    return new_name
