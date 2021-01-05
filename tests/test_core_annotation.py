# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:52:11 2020

@author: xavier.mouy
"""
#import pytest
import os
from ecosound.core.annotation import Annotation


def get_paths():
    """ Define paths of the test data."""
    paths = {'test_dir': os.path.dirname(os.path.realpath(__file__))}
    paths['data_dir'] = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'data'))
    paths['raven_annot_dir'] = os.path.join(paths['data_dir'],'Raven_annotations')
    paths['pamlab_annot_dir'] = os.path.join(paths['data_dir'],'PAMlab_annotations')
    paths['wav_files_dir'] = os.path.join(paths['data_dir'],'wav_files')

    # Raven annotations files
    raven_annot_files = []
    raven_annot_files.append({'filename': '67674121.181018013806.Table.1.selections.txt',
                              'annot_number': 773,
                              'duplicates': 0})
    raven_annot_files.append({'filename': 'AMAR173.4.20190916T061248Z.Table.1.selections.txt',
                              'annot_number': 1114,
                              'duplicates': 557})
    paths['raven_annot_files'] = raven_annot_files
    return paths


def test_len_is_0():
    """ Test len of annot is 0 upon instantiation. """
    annot = Annotation()
    assert len(annot) == 0
    return None


def test_from_raven_singlefile(fileidx=0):
    """ Test number of annotations when importing a single Raven file.

    Only use one file for this test defined by fileidx=0
    """
    paths = get_paths()
    annot = Annotation()
    annot.from_raven(os.path.join(paths['raven_annot_dir'],paths['raven_annot_files'][fileidx]['filename']), verbose=False)
    assert len(annot) == paths['raven_annot_files'][fileidx]['annot_number'] - paths['raven_annot_files'][fileidx]['duplicates']
    return None

def test_from_raven_singlefile_with_duplicates(fileidx=1):
    """ Test that Raven annotation duplicates are removed correctly. """
    paths = get_paths()
    annot = Annotation()
    annot.from_raven(os.path.join(paths['raven_annot_dir'], paths['raven_annot_files'][fileidx]['filename']), verbose=False)
    assert len(annot) == paths['raven_annot_files'][fileidx]['annot_number'] - paths['raven_annot_files'][fileidx]['duplicates']
    return None


def test_from_raven_dir():
    """ Test number of annotations when importing full folder (Raven). """
    paths = get_paths()
    total_annotations = 0
    for file in paths['raven_annot_files']:
        total_annotations += file['annot_number'] - file['duplicates']
    annot = Annotation()
    annot.from_raven(paths['raven_annot_dir'], verbose=False)
    assert len(annot) == total_annotations
    return None

# print(len(annot2))


