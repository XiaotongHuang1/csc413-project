# Leveraging Transformer Models to Explore the Relationship Between News Content and Stock Market Trends

*Authors: Wangzhang Wu, Pei Lin, and Xiaotong Huang*
*Department of Computer Science, University of Toronto*
*Date: April 17, 2024*

## Abstract

Machine learning technologies, especially within the financial sector, have emerged as both a critical and popular area of focus. Concurrently, the Transformer model has emerged as a notably powerful tool for sentiment analysis, demonstrating unprecedented capabilities in interpreting complex data. We introduce TCN + Transformer, a model that combines the Transformer with the Temporal Convolutional Networks (TCN), aiming to set a new benchmark in handling time-series data. This study delves into the innovative application of the Transformer model, leveraging its strengths to analyze vast datasets of news articles and historical stock price movements. Our aim is to harness this model’s predictive prowess to forecast future market trends. Through our investigation, we present a nuanced understanding of the interplay between news-driven sentiment and stock price trends, offering promising insights for predictive analytics in financial markets.

## File description
**train.py**: This script serves as the central execution point for our model's training pipeline. It encompasses several key components essential for the systematic development and evaluation of our model. The file includes:
- **Model Construction**: Defines and configures the neural network architecture tailored to our specific analytical requirements.
- **Data Loading**: Implements efficient mechanisms for ingesting and preprocessing data, ensuring it is formatted correctly for model training.
- **Model Training**: Orchestrates the training process, including parameter optimization and monitoring training progress to achieve optimal model performance.
- **Report Generation**: Automates the creation of detailed training reports, summarizing model performance metrics and key outcomes, facilitating an informed evaluation of the model's efficacy.
This script is designed to ensure a cohesive workflow from initial data handling to the final reporting stage, promoting reproducibility and efficiency in our model training endeavours.



## Introduction

When individuals read a news article regarding a corporation, it is common for them to speculate about its potential impact on the stock market and the company’s future. Frequently asked questions might include inquiries such as: ’What is the forecasted stock performance of the company?’ and ’Are there expectations for growth or decline?’ In reality, the interplay between financial markets and information dissemination has long been a subject of interest in the field of financial analytics. Both the general public and specialists—from researchers to corporations—are fascinated with this field, as all parties aim to predict stock market trends and secure financial profits.

After the "mighty" transformer model structure has been introduced [Vaswani et al., 2023], machine learning and natural language processing have opened new avenues for exploring context learning, enabling systems to understand, interpret, and generate human language in ways that were previously unattainable. This technological evolution has fostered the development of algorithms that can analyze vast amounts of text data, identifying patterns, sentiments, and nuanced meanings embedded within the language itself.

Therefore, we have decided to employ a transformer model to analyze news headlines and historical stock prices, aiming to predict market movements. Our goal is to explore whether the transformer model is capable of gaining a profound insight and accurately predicting market directions by learning patterns and nuances within the content and tone of news, as well as the history of stock prices. After researching various transformer architectures, we decided to choose the Bidirectional Encoder Representations from Transformers (BERT) model [Devlin et al., 2019] to explore. Due to BERT’s bidirectional nature, it possesses a more powerful capability for understanding context.

In addition to the transformer models, Temporal Convolutional Networks (TCNs) have also shown promise in the financial domain. Dai et al. [Dai et al., 2022] employed a TCN to predict the conditional probability of price changes in ultra-high frequency (UHF) stock price change data, with the addition of an attention mechanism to model the time-varying distribution. Their empirical research on the constituent stocks of the Chinese Shenzhen Stock Exchange 100 Index (SZSE 100) demonstrated that the TCN framework outperformed both GARCH family models and LSTM models in describing the dynamic process of UHF stock price change sequences. Furthermore, Wan et al. [Wan et al., 2019] proposed a novel Multivariate Temporal Convolution Network (M-TCN) model for multivariate time series forecasting, which showed significant improvements in prediction accuracy and robustness compared to traditional deep learning models such as LSTM and ConvLSTM. These studies highlight the potential of TCNs in capturing temporal dependencies and enhancing the accuracy of financial market predictions.

Motivated by the strengths of both TCNs and transformer models, we propose a novel architecture that combines the TCN’s ability to capture local temporal features with the transformer’s prowess in understanding global dependencies. This hybrid approach aims to provide a more comprehensive analysis of financial time series data, leveraging the complementary strengths of both models to improve prediction accuracy and reliability in financial market forecasting.

## Literature Review

### Influence of News on Stock Markets

The effect of the news on stock prices has been extensively studied, with researchers investigating how different types of news influence market behavior. For example, Tetlock [Tetlock, 2007] demonstrated that the sentiment of firm-specific news stories is predictive of stock price movements and trading volumes. Similarly, Bollen [Bollen et al., 2011] showed that public mood derived from Twitter feeds could be used to forecast stock market changes. These studies highlight the potential of textual information in predicting stock market trends.

### Social Media and Stock Markets

The advent of social media has further amplified the impact of news on financial markets. Platforms like Twitter and Facebook have become important sources of information, influencing investor sentiment and market movements. Luo [Luo et al., 2013] found that social media activity is a significant leading indicator of stock returns and trading volumes. This underscores the importance of incorporating social media data into financial analysis models.

### Deep Learning for Stock Prediction

Deep learning techniques, particularly Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks have shown promise in modelling time-series data and capturing temporal dependencies in stock market data. For instance, the use of LSTM networks has been demonstrated to improve the accuracy of stock price predictions by effectively handling long-term dependencies [Hochreiter and Schmidhuber, 1997

]. However, while RNNs and LSTMs excel at capturing sequential patterns, they may struggle with capturing long-range dependencies and understanding global context.

## Methodology

Our proposed model, TCN + Transformer, combines the strengths of TCNs and transformer models to enhance the prediction accuracy of stock market trends. The TCN component is responsible for capturing local temporal features in the data, while the transformer component focuses on understanding the global context and dependencies within the data. By integrating these two models, we aim to improve the model's ability to capture both short-term fluctuations and long-term trends in stock prices.

### TCN Architecture

The TCN component of our model consists of a series of dilated causal convolutional layers followed by a global average pooling layer. The dilated causal convolutional layers enable the model to capture local temporal features in the data, while the global average pooling layer aggregates these features to generate a global representation of the input sequence. This global representation is then passed to the transformer component for further processing.

### Transformer Architecture

The transformer component of our model consists of a stack of transformer layers, each comprising a multi-head self-attention mechanism followed by a feedforward neural network. The multi-head self-attention mechanism enables the model to capture dependencies between different parts of the input sequence, allowing it to understand the global context of the data. The feedforward neural network further refines the representation generated by the self-attention mechanism, producing a final representation that is used for prediction.

### Training

Our model is trained using a combination of news headlines and historical stock prices. The news headlines are tokenized and passed through the transformer component, which generates a representation of the headlines. This representation is then concatenated with the historical stock prices and passed through the TCN component, which generates a final representation that is used for prediction. The model is trained using a mean squared error loss function and optimized using the Adam optimizer.

## Results

We evaluate the performance of our model on a dataset consisting of news headlines and historical stock prices for a set of companies. We compare the performance of our model with that of a baseline LSTM model and a transformer-only model. Our results show that our proposed TCN + Transformer model outperforms both the baseline LSTM model and the transformer-only model, achieving higher accuracy in predicting stock market trends.

## Conclusion

In this study, we propose a novel TCN + Transformer model for predicting stock market trends using news headlines and historical stock prices. Our model combines the strengths of TCNs and transformer models to improve the prediction accuracy of stock market trends. Our results show that our proposed model outperforms both a baseline LSTM model and a transformer-only model, demonstrating the effectiveness of our approach. Overall, our study contributes to the growing body of research on using deep learning techniques for financial market prediction and provides insights into the potential of combining TCNs and transformer models for this task.
