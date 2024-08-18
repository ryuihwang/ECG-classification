model.eval()
probs = []
preds = []
labels = []
study_ids = []

with torch.no_grad():
    for inputs, target, study_id_batch in zip(eval_loader.dataset.tensors[0], eval_loader.dataset.tensors[1], study_ids_eval):
        inputs = inputs.unsqueeze(0).to(device) 
        target = target.to(device)
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        predictions = (probabilities > 0.1).astype(int)

        probs.extend(probabilities.flatten())
        preds.extend(predictions.flatten())
        labels.extend(target.cpu().numpy().flatten())

        study_ids.append(study_id_batch)

probs = np.array(probs)
preds = np.array(preds).astype(str)
labels = np.array(labels).astype(str)
study_ids = np.array(study_ids).astype(str)

results_df = pd.DataFrame({
    'study_id': study_ids,
    'PROB': probs,
    'PRED': preds,
    'LABEL': labels
})

print(results_df.head())

results_df.to_csv('new_directory/result/second_ex.csv', index=False)