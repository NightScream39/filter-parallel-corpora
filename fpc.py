#!/usr/bin/env python3
import argparse
import logging

import laser
import ruamel.yaml
from opusfilter.opusfilter import OpusFilter

yaml = ruamel.yaml.YAML()
logging.basicConfig(level=logging.INFO)
logging.getLogger('mosestokenizer.tokenizer.MosesTokenizer').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(prog='fpc',
    description='Filter Paralell Corpora (EN/RU)')

parser.add_argument('config', metavar='CONFIG', help='YAML configuration file')
parser.add_argument('--overwrite', '-o', help='overwrite existing output files', action='store_true')
parser.add_argument('--last', type=int, default=None, help='Last step to run')
parser.add_argument('--single', type=int, default=None, help='Run only the nth step')
parser.add_argument('--n-jobs', type=int, default=None,
    help='Number of parallel jobs when running score, filter and preprocess.')
parser.add_argument('--laser', '-l', help='use laserembeddings for filter', action='store_true')

args = parser.parse_args()

configuration = yaml.load(open('opus.yaml'))

if args.n_jobs is not None:
    configuration['common']['default_n_jobs'] = args.n_jobs

of = OpusFilter(configuration)

if args.single is None:
    of.execute_steps(overwrite=args.overwrite, last=args.last)
else:
    of.execute_step(args.single, overwrite=args.overwrite)

if args.laser:
    laser.filter_corpora_by_laser()