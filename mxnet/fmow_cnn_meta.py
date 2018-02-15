# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import data, fit, modelzoo
import mxnet as mx


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    meta = mx.sym.Variable('meta_data')
    net = mx.sym.Concat(*[net,meta])
    net = mx.sym.FullyConnected(data=net, name='feature1', num_hidden=2048)
    net = mx.sym.Activation(data=net, name='feature1_relu', act_type="relu")
    net = mx.sym.Dropout(data=net,p=0.5)
    net = mx.sym.FullyConnected(data=net, name='feature2', num_hidden=2048)
    net = mx.sym.Activation(data=net, name='feature2_relu', act_type="relu")
    net = mx.sym.Dropout(data=net,p=0.5)
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    parser.set_defaults(image_shape='3,299,299', num_epochs=8,
                        lr=.01, lr_step_epochs='6', wd=0, mom=0)

    args = parser.parse_args()

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if "densenet" not in args.pretrained_model:
        (prefix, epoch) = modelzoo.download_model(
           args.pretrained_model, os.path.join(dir_path, 'model'))
        if prefix is None:
            (prefix, epoch) = (args.pretrained_model, args.load_epoch)
    else:
        prefix = 'model/densenet-imagenet-169-0'
        epoch = 125
        print(prefix,epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)
    #epoch = 7
    #prefix = './model/imagenet11k-place365ch-resnet-152-cnn-meta-simplecut'
    #prefix = 'model/imagenet11k-place365ch-resnet-50-simple'
    #new_sym, new_args, aux_params = mx.model.load_checkpoint(prefix, epoch)
    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter,
            arg_params  = new_args,
            aux_params  = aux_params)
