import numpy as np
import pandas as pd
from datasets import Dataset
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments

df = pd.read_csv('../data/final_data_top5.csv') ## use your own customized dataset
df['headlines'] = df['headlines'].apply(lambda x: ast.literal_eval(x))
df['headlines'] = df['headlines'].apply(lambda x: ' [SEP] '.join(x))
df['headlines'] = "[CLS] " + df['headlines']
df['label'] = df['y']
# df['prices'] = df['prices'].apply(lambda x: x.replace("[", "").replace("]", ""))
# df['input'] = df.apply(lambda row: row['headlines'], axis=1)

df = df.drop('date', axis=1)
df = df.drop('prices', axis=1)
df = df.drop('y', axis=1)
# df = df.drop('headlines', axis=1)

# df['length'] = df['headlines'].apply(lambda x: len(x))
df.to_csv('../data/clean_input.csv', index=False)

print(df.columns)
print(df.head())

df_train, df_test, = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, stratify=df_train['label'],test_size=0.1, random_state=42)
print(df_train.shape, df_test.shape, df_val.shape)

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)

dataset_train = dataset_train.map(lambda e: tokenizer(e['headlines'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_val = dataset_val.map(lambda e: tokenizer(e['headlines'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_test = dataset_test.map(lambda e: tokenizer(e['headlines'], truncation=True, padding='max_length' , max_length=128), batched=True)

dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy' : accuracy_score(predictions, labels)}

args = TrainingArguments(
        output_dir = '../temp/',
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
)

trainer = Trainer(
        model=finbert,                         # the instantiated 🤗 Transformers model to be trained
        args=args,                  # training arguments, defined above
        train_dataset=dataset_train,         # training dataset
        eval_dataset=dataset_val,            # evaluation dataset
        compute_metrics=compute_metrics
)

trainer.train()

finbert.eval()
trainer.predict(dataset_test).metrics

trainer.save_model('finbert-sentiment/')