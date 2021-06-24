from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
import time

from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

total_start = time.time()
parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/decompensation/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--datasize',type=str,help='use small or large dataset',default='4')

parser.set_defaults(deep_supervision=False)
args = parser.parse_args()

print(args)

if args.small_part:
    args.save_every = 2**30

args_dict = dict(args._get_kwargs())
#args_dict['header'] = discretizer_header
args_dict['task'] = 'decomp'


# Build the model
print("==> using model {}".format(args.network))
model_start = time.time()

model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = "{}.bs{}{}{}.ts{}".format("" if not args.deep_supervision else ".dsup",
                                   args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
model.compile(optimizer=optimizer_config,
              loss='binary_crossentropy')
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*chunk([0-9]+).*", args.load_state).group(1))
model_end = time.time()
print("load model time:",(model_end-model_start)*1000)

if args.mode == 'test':

    # ensure that the code uses test_reader

    names = []
    ts = []
    labels = []
    predictions = []

    if args.deep_supervision:
        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'test'),
                                                                  listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                                  small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                      normalizer, args.batch_size,
                                                      shuffle=False, return_names=True)

        for i in range(test_data_gen.steps):
            print("\tdone {}/{}".format(i, test_data_gen.steps), end='\r')
            ret = next(test_data_gen)
            (x, y) = ret["data"]
            cur_names = np.array(ret["names"]).repeat(x[0].shape[1], axis=-1)
            cur_ts = ret["ts"]
            for single_ts in cur_ts:
                ts += single_ts

            pred = model.predict(x, batch_size=args.batch_size)
            for m, t, p, name in zip(x[1].flatten(), y.flatten(), pred.flatten(), cur_names.flatten()):
                if np.equal(m, 1):
                    labels.append(t)
                    predictions.append(p)
                    names.append(name)
        print('\n')
    else:
        data_start = time.time()
        test_reader = DecompensationReader(dataset_dir=os.path.join(args.data, ('test_'+args.datasize)),
                                           listfile=os.path.join(args.data, ('test_'+args.datasize+'_listfile.csv')))
        data_end = time.time()
        print("load data time:",(data_end-data_start)*1000)
        preprocess_start = time.time()
        discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

        if args.deep_supervision:
            discretizer_header = discretizer.transform(test_data_loader._data["X"][0])[1].split(',')
        else:
            discretizer_header = discretizer.transform(test_reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1] 

        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = 'decomp_ts{}.input_str:previous.n1e5.start_time:zero.normalizer'.format(args.timestep)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)
        test_data_gen = utils.BatchGen(test_reader, discretizer,
                                       normalizer, args.batch_size,
                                       None, shuffle=False, return_names=True)  # put steps = None for a full test
        preprocess_end =time.time()
        print("process data time:",(preprocess_end-preprocess_start)*1000)
        start_time = time.time()
        for i in range(test_data_gen.steps):
            print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
            ret = next(test_data_gen)
            x, y = ret["data"]
            cur_names = ret["names"]
            cur_ts = ret["ts"]
            
            x = np.array(x)
            pred = model.predict_on_batch(x)[:, 0]
            predictions += list(pred)
            labels += list(y)
            names += list(cur_names)
            ts += list(cur_ts)
    end_time = time.time()
    print("The inference time is ",(end_time-start_time)*1000)
    save_start =time.time()
    #metrics.print_metrics_binary(labels, predictions)
    path = os.path.join(args.output_dir, 'test_predictions', os.path.basename(args.load_state)) + '.csv'
    utils.save_results(names, ts, predictions, labels, path)
    total_end = time.time()
    print("save time is:",(total_end-save_start)*1000)
    print("total time is ",(total_end-total_start)*1000)
else:
    raise ValueError("Wrong value for args.mode")
