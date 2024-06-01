def train(epoch, criterion, model, optimizer, loader, device):
    
    total_loss = 0.0

    model.train()

    for batch_idx, (data, target) in enumerate(loader):
        
        optimizer.zero_grad()
        gt = target[:,1:]
        target = target[:,0:-1]
        data, target,gt = data.to(device), target.to(device),gt.to(device)
        
        output,h,c = model((data,target))

        #Reshape the output for CrossEntropy
        batch_size, sequence_length, num_classes = output.size()
        output = output.view(batch_size * sequence_length, num_classes)
        gt = gt.view(batch_size * sequence_length)

        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))

        total_loss += loss.item() 

    return total_loss / len(loader.dataset)