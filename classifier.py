#!/usr/bin/env python3

from utility import DataParser
from models import HMModel, PredictionResults

import sys
import pickle
import argparse
from datetime import datetime

import logging
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")



def parse_data(folder_path):
    data = DataParser(folder_path).parse()
    markers = [0, 2, 3, 5, 7, 9, 11, 13, 14, 16, 17, 19, 21, 23, 25, 27] # fingertips, ihand, ohand and iwr
    required_data = data.filter_markers(markers, keep=True) 
    return required_data


def train_model(data, n_components=10, n_mix=8, n_iter=315, downsample_step=7):
    model = HMModel(n_components=n_components, n_mix=n_mix, n_iter=n_iter, downsample_step=downsample_step, tol=1, h_type='velpos', verbose=True)
    model.train(data)
    print('Model successfully trained.')
    return model


def store_model(model, name):
    with open(f'Pretrained/{name}.pkl', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved model to Pretrained/{name}.pkl")

def load_model(name):
    with open(f'Pretrained/{name}.pkl', 'rb') as f:
        return pickle.load(f)
    print(f"Loaded model Pretrained/{name}.pkl")


# subcommand train
def train(args):
    data = parse_data(args.filepath)
    model = train_model(data, n_components=args.n_components, n_mix=args.n_mix, n_iter=args.n_iter, downsample_step=args.downsample_step)
    name = args.name if args.name else f"model-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    store_model(model, name)


# subcommand classify
def classify(args):
    model = load_model(args.model)
    if args.labeled:
        test = parse_data(args.filepath)
        results = PredictionResults()
        i = 1
        for cls in test:
            for mocap in test[cls]:
                pred = model.predict(mocap)[0]
                results.append(pred, cls.name)
                print(i, pred.name)
                i += 1
        print(f"Overall accuracy: {100*results.accuracy}%", file=sys.stderr)

    else:
        # single input sequence
        mocap = DataParser.parse_single(args.filepath)
        pred = model.predict(mocap)[0]
        print(pred.name)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser('train', help='Train a HMM-based model and store it to disk.')
parser_train.set_defaults(func=train)
parser_train.add_argument('--name', type=str)
parser_train.add_argument('--n_components', default=10, type=int)
parser_train.add_argument('--n_mix', default=8, type=int)
parser_train.add_argument('--n_iter', default=305, type=int)
parser_train.add_argument('--downsample_step', default=7, type=int)
parser_train.add_argument('filepath', default="Data Split/Train-set", type=str)

parser_classify = subparsers.add_parser('classify', help='Use a pretrained model to classify an input sequence or a collection of input sequences.')
parser_classify.set_defaults(func=classify)
parser_classify.add_argument('-m', '--model', default='best', type=str)
parser_classify.add_argument('--labeled', action='store_true')
parser_classify.add_argument('filepath', type=str)

args = parser.parse_args()
args.func(args) # call subcommand function
