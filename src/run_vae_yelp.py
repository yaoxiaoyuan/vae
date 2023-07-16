# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:38:07 2020

"""
import sys
from config import parse_yelp_args
from train import train,eval_model
from utils import build_logger
from sample import sample_text

def main(arguments):
    """
    """
    args = parse_yelp_args(arguments)
    logger = build_logger()
    if args.train == True:
        train(args, logger)
    elif args.sample == False:
        eval_model(args, logger)
    else:
        sample_text(args, logger)

if __name__ == "__main__":
    main(sys.argv[1:])