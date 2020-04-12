import argparse
import os
import torch
import utilities
import json

parser = argparse.ArgumentParser()

parser.add_argument('input', action='store',
                    help='Path to input image')

parser.add_argument('checkpoint', action='store',
                    help='Path to model checkpoint')

parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    help='Number of top predictions to be returned',
                    default=1, type=int)

parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    help='Path to file with category names',
                    default='cat_to_name.json')

parser.add_argument('--gpu', action='store_true',
                    dest='use_gpu',
                    help='Use GPU for training',
                    default=False)

def main():
    
    arguments = parser.parse_args()
    
    if not os.path.exists(arguments.input):
        print('Error: Given input file does not exist')
        return
    
    if not os.path.exists(arguments.checkpoint):
        print('Error: Given checkpoint file does not exist')
        return
    
    if not os.path.exists(arguments.category_names):
        print('Error: Given category to name mapping file does not exist')
        return
    
    if arguments.use_gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print('Error: System does not support CUDA')
            return
        
    model = torch.load(arguments.checkpoint)
    top_p, top_class = utilities.predict(arguments.input, model, device, topk=arguments.top_k)
    
    with open(arguments.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    top_label = list(map(cat_to_name.get, top_class))
    predictions = dict(zip(top_label, top_p))
    
    print('Neural network predicted the following categories with corresponding probabilities:\n\r', predictions)
        
if __name__ == '__main__':
    main()