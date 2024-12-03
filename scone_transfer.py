import argparse
import random
from typing import List
from tqdm import trange
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
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

def get_transfer_dataset(scone : pd.DataFrame, transfer : str, shuffle : bool = True):
    inputs = []
    labels = []
    if transfer == 'lex_rel':    
        for c in SCONE_CATEGORIES:
            inputs += scone[f'premise_{c}'].tolist()
            # always output lexical relation as if there was no negation
            labels += scone[f'label_no_negation'].tolist()
    elif transfer == 'neg_one':
        for c in ['one_scoped_one_not_scoped', 'two_not_scoped', 'two_scoped']:
            inputs += scone[f'premise_{c}'].tolist()
            # entailment if first negation is scoped, neutral otherwise
            if c == 'one_scoped_one_not_scoped' or c == 'two_scoped':
                labels += ['entailment'] * len(scone[f'label_{c}'])
            else: # two_not_scoped
                labels += ['neutral'] * len(scone[f'label_{c}'])
    elif transfer == 'neg_two':
        for c in ['one_scoped_one_not_scoped', 'two_not_scoped', 'two_scoped']:
            inputs += scone[f'premise_{c}'].tolist()
            # entailment if second negation is scoped, neutral otherwise
            if c == 'one_scoped_one_not_scoped' or c == 'two_not_scoped':
                labels += ['neutral'] * len(scone[f'label_{c}'])
            else: # two_scoped
                labels += ['entailment'] * len(scone[f'label_{c}'])
    elif transfer == 'neg_count':
        for c in SCONE_CATEGORIES:
            inputs += scone[f'premise_{c}'].tolist()
            # entailment if 0, neutral if 1, contradiction if 2 scoped
            if c == 'no_negation' or c == 'two_not_scoped' or c == 'one_not_scoped':
                labels += ['entailment'] * len(scone[f'label_{c}'])
            elif c == 'one_scoped' or c == 'one_scoped_one_not_scoped':
                labels += ['neutral'] * len(scone[f'label_{c}'])
            else: # two_scoped
                labels += ['contradiction'] * len(scone[f'label_{c}'])

    dataset = list(zip(inputs, labels))
    if shuffle:
        random.shuffle(dataset)

    return zip(*dataset)

def train(
    model : AutoModelForSequenceClassification, 
    tokenizer : AutoTokenizer,
    scone : pd.DataFrame, 
    transfer : str,
    num_epochs : int = 1,
    learning_rate : float = 5e-6,
    batch_size : int = 8
):
    device = model.device

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_x, train_labels = get_transfer_dataset(scone, transfer, shuffle=True)
    losses = []
    for epoch in range(num_epochs):
        with trange(0, len(train_x), batch_size, desc=f'Epoch {epoch+1}') as pbar:
            for b in pbar:
                optimizer.zero_grad()
                batch_x = train_x[b:b+batch_size]
                batch_labels = train_labels[b:b+batch_size]
                batch_labels = torch.tensor([LABEL_NAMES.index(l) for l in batch_labels])

                inputs = tokenizer(batch_x, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs.to(device))
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, batch_labels.to(device))
                loss.backward()
                optimizer.step()
                pbar.set_description(f'Loss: {loss.item():.2f}')
                losses.append(loss.item())
    return losses

def predict(
    model : AutoModelForSequenceClassification, 
    tokenizer : AutoTokenizer,
    scone : pd.DataFrame, 
    transfer : str,
    batch_size : int = 8,
):
    device = model.device

    model.eval()
    test_x, _ = get_transfer_dataset(scone, transfer, shuffle=False)

    predictions = []
    for b in trange(0, len(test_x), batch_size, desc='Eval'):
        batch_x = test_x[b:b+batch_size]
        inputs = tokenizer(batch_x, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs.to(device))
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        predictions += preds
    
    predictions = [LABEL_NAMES[p] for p in predictions]

    return predictions

def plot_losses(losses : List[float], save_path : str):
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(save_path)

def main(
    model_name_or_path : str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    seed : int = 0,
    num_epochs : int = 1,
    batch_size : int = 16
):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, device_map=device)

    print('Loading data...')
    scone_train = load_scone('train')
    scone_test = load_scone('test')


    for transfer in ['lex_rel', 'neg_one', 'neg_two', 'neg_count']:

        print(f'Training ({transfer})...')
        losses = train(model, tokenizer, scone_train, transfer, num_epochs=num_epochs, learning_rate=5e-6, batch_size=batch_size)

        print(f'Predicting ({transfer})...')
        predictions = predict(model, tokenizer, scone_test, transfer, batch_size=batch_size)

        print(f'Saving results to results_{model_name_or_path}_transfer_{transfer}.csv...')
        results = pd.DataFrame(predictions, columns=SCONE_CATEGORIES)
        results.to_csv(f'results_{model_name_or_path}_transfer_{transfer}.csv', index=False)

        print(f'Saving loss plot to loss_{model_name_or_path}_transfer_{transfer}.png...')
        plot_losses(losses, f'loss_{model_name_or_path}_transfer_{transfer}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(
        model_name_or_path=args.model_name_or_path,
        seed=args.seed,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )