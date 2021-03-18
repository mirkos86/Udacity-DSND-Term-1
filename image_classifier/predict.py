import argparse
import os
import predict_utils
from predict_utils import load_flower_checkpoint, predict

parser=argparse.ArgumentParser(description='Load a trained image classifier through a checkpoint and predict the top 5 most likely classes')

parser.add_argument(action='store',dest='filepath',type=str,
help='Indicate the path to the image to be processed')

parser.add_argument('--cat',action='store',dest='cat_path',type=str,default=None,
help='Indicate the path to a possible .json file mapping categories to flower names')

parser.add_argument('--classes',action='store',dest='classes',type=int,default=1,
help='Specify how many of the most likely classes you would like to see displayed')

parser.add_argument('--check_path',action='store',type=str,dest='check_path',default='checkpoint_dir/flower_checkpoint.pth',
                    help='Specify the checkpoint file path')

parser.add_argument('--gpu',action='store_true',default=False,
help='Specify if you want to make the model run on a gpu (if it is available)')

# Read the command line instructions and parse them
args=parser.parse_args()

#########################################
# PARSING IS OVER; THE PREDICTION BEGINS#
#########################################

#First things firt: we load the saved checkpoint
flower_model = load_flower_checkpoint(args.check_path)

# Secondy, we apply the "predict" function, which incorporates the image-preprocessing function, 
# to predict the classes top k most probable classes, along with their indices

probs, classes = predict(args.filepath, flower_model, args.gpu, topk=args.classes)

# Here we check if the user supplied a path to a .json file containing a map from classes to names.
# If so, we turn this map into an index-to-name mapping and print the names of the classesinstead of their labels
# Otherwise, we go with the labels

if args.cat_path == None:
    print(probs)
    print(classes)
    
else:
    import json
    with open(args.cat_path, 'r') as f:
        # 1. We first open the .json file and import the categories-to-name mapping. We then sort it.
        cat_to_name = json.load(f) 
        idx_to_labels = {idx: cat_to_name[cat] for cat, idx in flower_model.class_to_idx.items()}
        
    print(probs)
    print(list(map(idx_to_labels.get,classes)))



