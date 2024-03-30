

from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')



nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
results = nlp([ """Citadel's Ken Griffin says the Fed shouldn't cut too quickly, citing tailwinds supporting inflation Jefferies analyst challenges Trump’s claim that Meta’s Facebook is ‘enemy of the people’ Stocks making the biggest moves midday: Southwest, Oracle, 3M, New York Community Bancorp and more Stocks making the biggest moves before the bell: Oracle, Kohl's, Coinbase, Southwest and more Xiaomi is set to launch its electric car on March 28""",
               'there is a shortage of capital, and we need extra financing.',
              'formulation patents might protect Vasotec to a limited extent.'])

print(results)