#! /usr/bin/env python3
import argparse
import sys
import pprint as pp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='provide arguments for experiment')

    # agent parameters
    parser.add_argument('--model-lr', help='network learning rate', default=0.00001)
    parser.add_argument('--lamda', help='network regularization parameter', default=0.01)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=32)
    parser.add_argument('--max-epoch', help='No. of epoch', default=10)
    parser.add_argument('--dropout', help='dropout prob.', default=0.5)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--model-dir', help='directory for storing the trained models',
                        default='./defects_cls_model_dumps/')
    parser.add_argument('--model-results', help='directory for storing results of models',
                        default='./defects_cls_output/')
    parser.add_argument('--data-path', help='directory for dataset', default='../dataset_dir/')

    parser.add_argument('--train-mode', action='store_true', help='Training mode')
    parser.add_argument('--semi-sup-mode', action='store_true', help='Semi-supervised enhancement mode')
    parser.add_argument('--eval-mode', action='store_true', help='Evaluation mode')
    parser.add_argument('--label-method', help="Unlabelled label assignment method",
                        choices=["knn", "baseline", "label_prop"], default="knn")
    parser.add_argument('--verbose', action="store_true", help='verbose mode')

    args = vars(parser.parse_args())
    pp.pprint(args)
    pp.pprint(args, stream=sys.stderr)
    print(bool(args["train_mode"]))
    print(bool(args["semi_sup_mode"]))
    print(bool(args["eval_mode"]))

