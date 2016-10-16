# pylint: disable=missing-docstring, import-error
import os
import subprocess
from operator import itemgetter
import pandas as pd


def xgbfi(bst, feature_names, fmap='xgb.fmap', fdump='xgb.dump', **kwargs):
    create_feature_map(feature_names, fmap)
    dump_model(bst, fdump, fmap)
    kwargs['dump_file'] = fdump
    return call_xgbfi(**kwargs)


def get_importance(bst, feature_names, fmap='xgb.fmap', as_pandas=True,
                   fscore=True):
    create_feature_map(feature_names, fmap)
    if fscore:
        imp = bst.get_fscore(fmap=fmap)
    else:
        imp = bst.get_score(fmap=fmap)
    imp = sorted(imp.items(), key=itemgetter(1), reverse=True)
    if as_pandas:
        imp = pd.DataFrame(
            {'feature': [val[0] for val in imp],
             'importance': [val[1] for val in imp]})
        imp['normalized'] = imp.importance / imp.importance.sum()
    return imp


def call_xgbfi(dump_file='xgb.dump', max_depth=4, max_deepening=-1,
               max_trees=1000, topk=100, sort_by='Gain',
               out_file='XgbFeatureInteractions', max_histograms=10,
               path='git/xgbfi/bin/XgbFeatureInteractions.exe'):
    # pylint: disable=too-many-arguments, too-many-locals
    home = os.environ['HOME']
    xgbfi_path = os.path.join(home, path)
    if not os.path.exists(xgbfi_path):
        raise ValueError('%s does not exist' % xgbfi_path)

    dump_file_arg = '-m %s' % dump_file
    max_depth_arg = '-d %d' % max_depth
    max_deepening_arg = '-g %d' % max_deepening
    max_trees_arg = '-t %d' % max_trees
    topk_args = '-k %d' % topk
    sort_by_arg = '-s %s' % sort_by
    out_file_arg = '-o %s' % out_file
    max_histograms_arg = '-h %d' % max_histograms

    subprocess.call([
        'mono', xgbfi_path,
        dump_file_arg, max_depth_arg, max_deepening_arg, max_trees_arg,
        topk_args, sort_by_arg, out_file_arg, max_histograms_arg])

    spreadsheet = '%s.xlsx' % out_file
    result = {}
    for i in range(max_depth):
        sheetname = 'Interaction Depth %d' % i
        name = '_'.join(sheetname.lower().split(' ')[1:])
        result[name] = pd.read_excel(spreadsheet, sheetname=sheetname)
    result['leaf'] = pd.read_excel(spreadsheet, sheetname='Leaf Statistics')
    return result


def create_feature_map(feature_names, fmap='xgb.fmap'):
    outfile = open(fmap, 'w')
    for i, feat in enumerate(feature_names):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def dump_model(bst, fdump='xgb.dump', fmap='xgb.fmap'):
    bst.dump_model(fdump, fmap, with_stats=True)
