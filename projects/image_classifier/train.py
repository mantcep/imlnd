import argparse
import os
import trainfunctions
import torch
from torchvision import models

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store',
                    help='Set directory containing data')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory',
                    help='Set directory to save checkpoints',
                    default=os.getcwd())

parser.add_argument('--arch', action='store',
                    dest='architecture',
                    help='Set neural network architecture',
                    default='vgg16')

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    help='Set learning rate',
                    default=0.001, type=float)

parser.add_argument('--hidden_units', action='append',
                    dest='hidden_units',
                    help='Append a hidden layer with the specified number of nodes',
                    default=[], type=int)

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    help='Set number of epochs',
                    default=5, type=int)

parser.add_argument('--gpu', action='store_true',
                    dest='use_gpu',
                    help='Use GPU for training',
                    default=False)

parser.add_argument('--version', action='version',
                    version='%(prog)s 0.1')


def main():
    
    arguments = parser.parse_args()
    
    if not os.path.exists(arguments.data_dir):
        print('Error: Given data directory folder does not exist')
        return
    
    if not os.path.exists(arguments.save_directory):
        print('Error: Given save directory folder does not exist')
        return
    
    device = 'cpu'
    if arguments.use_gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print('Error: System does not support CUDA')
            return
    
    try:
        model = getattr(models, arguments.architecture)(pretrained=True)
    except:
        print('Error: Given architecture is not found')
        return
    
    try:
        # This method for densenet pretrained models
        in_features = model.classifier.in_features
    except:
        try:
            # This method for vgg pretrained models
            in_features = model.classifier[0].in_features
        except:
            try:
                # This method for alexnet pretrained models
                in_features = model.classifier[1].in_features
            except:
                print('Error: Unable to extract number of inputs features for classifier')
                return
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    if not arguments.hidden_units:
        arguments.hidden_units = [1024, 256]    # Defaulting the hidden layers of none given
    
    model.classifier = trainfunctions.create_classifier(arguments.hidden_units, in_features)
    
    model = trainfunctions.train_network(model, device, arguments.data_dir, arguments.epochs, arguments.learning_rate)
    trainfunctions.test_network(model, device, arguments.data_dir)
    
    torch.save(model, os.path.join(arguments.save_directory, 'chckpnt.pth'))
    
if __name__ == '__main__':
    main()