def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def train(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.float()
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_probs, val_labels = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = targets.float()
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
            val_probs.extend(probabilities.flatten())
            val_labels.extend(targets.cpu().numpy().flatten())
    
    optimal_cutoff = find_optimal_cutoff(val_labels, val_probs)
    print(f'Optimal cutoff: {optimal_cutoff}')
    
    all_preds = (np.array(val_probs) > optimal_cutoff).astype(int)
    val_loss /= len(val_loader)
    return val_loss, val_probs, val_labels, all_preds

def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = threshold[ix]
    return best_threshold