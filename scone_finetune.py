import argparse
import random
from tqdm import trange
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = ["entailment", "neutral", "contradiction"]
SCONE_CATEGORIES = ['no_negation', 'one_not_scoped', 'one_scoped_one_not_scoped', 'one_scoped', 'two_not_scoped', 'two_scoped']

def load_scone(split : str = 'train'):
    scone_no_negation = pd.read_csv(f"ScoNe/scone_nli/{split}/no_negation.csv", index_col=0)
    one_not_scoped = pd.read_csv(f"ScoNe/scone_nli/{split}/one_not_scoped.csv", index_col=0)
    one_scoped_one_not_scoped = pd.read_csv(f"ScoNe/scone_nli/{split}/one_scoped_one_not_scoped.csv", index_col=0)
    one_scoped = pd.read_csv(f"ScoNe/scone_nli/{split}/one_scoped.csv", index_col=0)
    two_not_scoped = pd.read_csv(f"ScoNe/scone_nli/{split}/two_not_scoped.csv", index_col=0)
    two_scoped = pd.read_csv(f"ScoNe/scone_nli/{split}/two_scoped.csv", index_col=0)
    scone = one_scoped.rename(columns={
        'sentence1': 'premise_one_scoped',
        'sentence2': 'hypothesis_one_scoped',
        'gold_label': 'label_one_scoped'
    })

    scone['premise_no_negation'] = scone_no_negation['sentence1_edited']
    scone['hypothesis_no_negation'] = scone_no_negation['sentence2_edited']
    scone['label_no_negation'] = scone_no_negation['gold_label_edited']

    scone['premise_one_not_scoped'] = one_not_scoped['sentence1_edited']
    scone['hypothesis_one_not_scoped'] = one_not_scoped['sentence2_edited']
    scone['label_one_not_scoped'] = one_not_scoped['gold_label_edited']

    scone['premise_one_scoped_one_not_scoped'] = one_scoped_one_not_scoped['sentence1_edited']
    scone['hypothesis_one_scoped_one_not_scoped'] = one_scoped_one_not_scoped['sentence2_edited']
    scone['label_one_scoped_one_not_scoped'] = one_scoped_one_not_scoped['gold_label_edited']

    scone['premise_two_not_scoped'] = two_not_scoped['sentence1_edited']
    scone['hypothesis_two_not_scoped'] = two_not_scoped['sentence2_edited']
    scone['label_two_not_scoped'] = two_not_scoped['gold_label_edited']

    scone['premise_two_scoped'] = two_scoped['sentence1_edited']
    scone['hypothesis_two_scoped'] = two_scoped['sentence2_edited']
    scone['label_two_scoped'] = two_scoped['gold_label_edited']

    return scone

def get_dataset(scone : pd.DataFrame, shuffle : bool = True):
    hypotheses = []
    premises = []
    labels = []
    for c in SCONE_CATEGORIES:
        premises += scone[f'premise_{c}'].tolist()
        hypotheses += scone[f'hypothesis_{c}'].tolist()
        labels += scone[f'label_{c}'].tolist()

    dataset = list(zip(premises, hypotheses, labels))
    if shuffle:
        random.shuffle(dataset)

    return zip(*dataset)

def train(
    model : AutoModelForSequenceClassification, 
    tokenizer : AutoTokenizer,
    scone : pd.DataFrame, 
    num_epochs : int = 1,
    learning_rate : float = 5e-6,
    batch_size : int = 8
):
    device = model.device

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_p, train_h, train_labels = get_dataset(scone, shuffle=True)
    for epoch in range(num_epochs):
        with trange(0, len(train_p), batch_size, desc=f'Epoch {epoch+1}') as pbar:
            for b in pbar:
                optimizer.zero_grad()
                batch_p = train_p[b:b+batch_size]
                batch_h = train_h[b:b+batch_size]
                batch_labels = train_labels[b:b+batch_size]
                batch_labels = torch.tensor([LABEL_NAMES.index(l) for l in batch_labels])

                inputs = tokenizer(batch_p, batch_h, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs.to(device))
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, batch_labels.to(device))
                loss.backward()
                optimizer.step()
                pbar.set_description(f'Loss: {loss.item():.2f}')

def predict(
    model : AutoModelForSequenceClassification, 
    tokenizer : AutoTokenizer,
    scone : pd.DataFrame, 
    batch_size : int = 8,
):
    device = model.device

    model.eval()
    test_p, test_h, _ = get_dataset(scone, shuffle=False)

    predictions = []
    for b in trange(0, len(test_p), batch_size, desc='Eval'):
        batch_p = test_p[b:b+batch_size]
        batch_h = test_h[b:b+batch_size]
        inputs = tokenizer(batch_p, batch_h, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs.to(device))
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        predictions += preds
    
    predictions = [LABEL_NAMES[p] for p in predictions]
    predictions = np.array(predictions).reshape(len(SCONE_CATEGORIES), -1).T

    return predictions

def main(
    seed : int = 0,
    num_epochs : int = 1,
    batch_size : int = 8
):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('Loading model...')
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map=device)

    print('Loading data...')
    scone_train = load_scone('train')
    scone_test = load_scone('test')

    print('Training...')
    train(model, tokenizer, scone_train, num_epochs=num_epochs, learning_rate=5e-6, batch_size=batch_size)

    print('Predicting...')
    predictions = predict(model, tokenizer, scone_test, batch_size=batch_size)


    print(f'Saving results to results_seed_{seed}_epochs_{num_epochs}.csv...')
    results = pd.DataFrame(predictions, columns=SCONE_CATEGORIES)
    results.to_csv(f'results_seed_{seed}_epochs_{num_epochs}.csv', index=False)

    print(f'Saving model to model_seed_{seed}_epochs_{num_epochs}')
    model.save_pretrained(f'model_seed_{seed}_epochs_{num_epochs}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(
        seed=args.seed,
        num_epochs=args.num_epochs
    )