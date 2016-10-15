# pylint: disable=missing-docstring, import-error
from operator import itemgetter
import pandas as pd


def create_feature_map(features, fmap='xgb.fmap'):
    outfile = open(fmap, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(bst, features, fmap='xgb.fmap', as_pandas=True, fscore=True):
    create_feature_map(features, fmap)
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


def dump_model(bst, fdump='xgb.dump', fmap='xgb.fmap'):
    bst.dump_model(fdump, fmap, with_stats=True)


# TODO: Criar funcao que chame xgbfi
