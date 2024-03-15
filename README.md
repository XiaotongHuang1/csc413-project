# Leveraging Transformer Models to Explore the Relationship Between News Content and Stock Market Trends

## Introduction
Our project combines financial analytics with advanced machine learning, using Transformer models like BERT and GPT-3 to analyze financial news and predict stock market trends. This approach aims to improve investment insights by identifying patterns in large datasets, setting new standards in financial predictive analysis, and emphasizing machine learning's role in financial decision-making.

## Background & Related Work
The interplay between financial markets and information dissemination has long been a subject of interest in the field of financial analytics. Recent advancements in machine learning and natural language processing have opened new avenues for exploring this relationship, particularly in the context of news and social media's impact on stock market dynamics. This section reviews the relevant literature and sets the stage for our project, which aims to harness the power of Transformer models for financial concept analysis and business model prediction.

### Influence of News on Stock Markets
The effect of news on stock prices has been extensively studied, with researchers investigating how different types of news influence market behavior. For example, Tetlock et al. (2007) demonstrated that the sentiment of firm-specific news stories is predictive of stock price movements and trading volumes. Similarly, Bollen et al. (2011) showed that public mood derived from Twitter feeds could be used to forecast stock market changes. These studies highlight the potential of textual information in predicting stock market trends.

### Social Media and Stock Markets
The advent of social media has further amplified the impact of news on financial markets. Platforms like Twitter and Facebook have become important sources of information, influencing investor sentiment and market movements. Luo et al. (2013) found that social media activity is a significant leading indicator of stock returns and trading volumes. This underscores the importance of incorporating social media data into financial analysis models.

### Deep Learning for Stock Prediction
Deep learning techniques, particularly Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, have shown promise in modeling time-series data and capturing temporal dependencies in stock market data. For instance, the use of LSTM networks has been demonstrated to improve the accuracy of stock price predictions by effectively handling long-term dependencies (Hochreiter & Schmidhuber, 1997). More recently, attention mechanisms have been introduced to enhance the model's ability to focus on relevant information, further improving prediction accuracy.

A novel approach to address the limitations of traditional deep learning models in stock trend prediction is the Knowledge-Driven Temporal Convolutional Network (KDTCN) proposed by Deng et al. (2020). This model incorporates background knowledge, news events, and price data to tackle the problem of predicting stock trends with abrupt changes. By integrating structured knowledge graphs and textual news into a Temporal Convolution Network (TCN), KDTCN not only improves the accuracy of predictions but also provides explanations for the predictions, making it a significant advancement in the field of stock prediction.

### Transformer Models in Finance
Transformer models, such as BERT (Bidirectional Encoder Representations from Transformers) and GPT-3 (Generative Pre-trained Transformer 3), have revolutionized natural language processing tasks. Their ability to capture contextual relationships in text data makes them well-suited for analyzing financial news and reports. Our project aims to leverage these models to extract financial concepts and predict business models with high precision, offering valuable insights to stakeholders in the financial sector.

## Data Processing
We utilize three primary sources of data to analyze the impact of news and structured knowledge on stock market trends:
1. **Time-series Price Data:** Our price dataset comprises daily value records of the Dow Jones Industrial Average (DJIA) index, spanning from Jan 1st, 2000, to Mar 15th, 2024. We obtained this data directly from cnbc.com Finance. To ensure consistency and accuracy in our analysis, we performed a thorough cleaning process to remove any discrepancies due to bank holidays. Additionally, we aligned the stock price data with the corresponding financial news by date to maintain the temporal coherence of our dataset.
2. **Textual News Data:** The news dataset consists of historical news headlines sourced from the Reddit WorldNews Channel. For each trading day, we selected the top 25 news headlines based on the votes of Reddit users. This dataset covers the same period as the price data, from Jan 1st, 2000, to Mar 15th, 2024. To prepare the textual data for analysis, we conducted preprocessing steps such as tokenization, removal of stop words, and stemming to reduce the noise and improve the relevance of the news content for our models.
3. **Structured Knowledge Data:** Our structured knowledge data is derived from two widely used open knowledge graphs, Freebase and Wikidata. We constructed a sub-graph from these sources, containing a total of 64,958 entities and 716 relations. This structured data provides a rich semantic context for the events mentioned in the news headlines, enabling us to enhance the interpretability and accuracy of our models.

## Architecture
The full structure is introduced in our paper, which also includes a figure description under the introduction. We basically substitute the deep learning layer with a transformer layer. The project's transformer layer, which is the core part, will most likely be an encoder-only transformer structure (Bert-like model). The bidirectional model has advantages in contextual understanding, which will be helpful for exploring the underlying meanings in the news articles. The model will take textual news data as its encoder inputs and outputs a binary classifier (true or false) to indicate whether the stock has an increasing or decreasing trend. We might fine-tune a pre-trained Bert model: ProsusAI/finbert · Hugging Face.

## Baseline Model
For the baseline model, we are considering using a decision tree or KNN. We will also be comparing the following models: Autoregressive integrated moving average (ARIMA); Long Short-Term Memory (LSTM); Convolutional Neural Network (CNN); and Temporal Convolutional Network (TCN). Their accuracies are introduced in our paper, and we will use them as a reference. If our model works well, we will further improve our model to compete with advanced TCN models.
