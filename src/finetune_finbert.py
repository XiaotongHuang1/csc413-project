import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments

df = pd.read_csv('../data/final_dataset/finetune_data.csv')  ## use your own customized dataset
df['label'] = df['y'].apply(lambda x: 2 if x == 0 else 1)
df = df.drop('y', axis=1)

df_train, df_test, = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, stratify=df_train['label'], test_size=0.1, random_state=42)
print(df_train)

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)

dataset_train = dataset_train.map(
    lambda e: tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_val = dataset_val.map(lambda e: tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=128),
                              batched=True)
dataset_test = dataset_test.map(
    lambda e: tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=128), batched=True)

dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(predictions, labels)}


args = TrainingArguments(
    output_dir='../temp/',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=20,
    weight_decay=0.002,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

trainer = Trainer(
    model=finbert,  # the instantiated 🤗 Transformers model to be trained
    args=args,  # training arguments, defined above
    train_dataset=dataset_train,  # training dataset
    eval_dataset=dataset_val,  # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()

finbert.eval()
trainer.predict(dataset_test).metrics

trainer.save_model('finbert-sentiment/')
