def cross_validate(model, X, y, num_folds=5, num_epochs=10, patience=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    train_losses = []
    val_losses = []
    all_labels = []
    all_preds = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f'Fold {fold + 1}/{num_folds}')
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        train_dataset = TensorDataset(torch.Tensor(X_train).unsqueeze(1), torch.Tensor(y_train).unsqueeze(1))
        val_dataset = TensorDataset(torch.Tensor(X_val).unsqueeze(1), torch.Tensor(y_val).unsqueeze(1))
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        model.apply(weights_init)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCEWithLogitsLoss()
        
        early_stopping_counter = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss = train(epoch, model, train_loader, optimizer, criterion, device)
            val_loss, val_probs, val_labels, val_preds = validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
        
        all_labels.extend(val_labels)
        all_preds.extend(val_preds)
    
    plot_loss(train_losses, val_losses)

    plot_confusion_matrix(all_labels, all_preds)