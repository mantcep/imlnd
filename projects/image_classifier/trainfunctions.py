import torch
from torch import nn, optim
from torchvision import transforms, datasets
from collections import OrderedDict

def create_loaders(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    class_to_idx = train_data.class_to_idx

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, class_to_idx

def create_classifier(hidden_units, in_features):
    
    od = OrderedDict()
    fin = in_features
    for i, hidden_unit in enumerate(hidden_units):
        fout = hidden_unit
        od.update({f'fc{i}': nn.Linear(fin, fout), f'relu{i}': nn.ReLU(), f'drop{i}': nn.Dropout()})
        fin = hidden_unit
    od.update({f'fc{i+1}': nn.Linear(fin, 102), 'output': nn.LogSoftmax(dim=1)})    # Hard-coding output to 102 categories
    
    classifier = nn.Sequential(od)
    
    return classifier

def train_network(model, device, data_dir, epochs, lr):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    print_every = 100
    steps = 0
    
    trainloader, validloader, __, class_to_idx = create_loaders(data_dir)
    
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {validation_accuracy/len(validloader):.3f}")
                model.train()

        print(f"Training loss: {running_loss/len(trainloader)}")
    
    model.class_to_idx = class_to_idx
    return model
        
def test_network(model, device, data_dir):
    test_loss = 0
    test_accuracy = 0
    criterion = nn.NLLLoss()
    
    __, __, testloader, __ = create_loaders(data_dir)
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {test_accuracy/len(testloader):.3f}")