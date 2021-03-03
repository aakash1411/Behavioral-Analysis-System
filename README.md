# Behavioral Analysis System: An end-to-end user depression detection system.
## Business Problem
Over the years, the number of suicide cases in the United States of America has exponentially increased and has been the leading cause of death throughout different age groups and genders. This is due to the lack of awareness related to mental health issues and moreover, the lack of access to care. This Scenario has developed in recent years, where a lot of depressed patients or patients who have been through trauma seek for programs that provide them behavioral healthcare. This is not enough as it is really difficult for the care providers to assess the time of crisis a patient can be in, and fail to provide care when it is needed the most in a time of relapses. Credits: [The State Of Mental Health In America](https://www.mhanational.org/issues/state-mental-health-america "Mental Health America")
## Problem Statement
Care providers enroll patients in a program where they maintain their thoughts and program related details in a digital journal. The intent here is to follow up on the digital journal and process their thoughts to understand their severity of depression and if there is any immediate threat to themselves or others around them.
## Sources
  * [Sentiment140](http://www.sentiment140.com/) - A Twitter Sentiment Analysis Tool.
  * [Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
  * [SpaCy](https://spacy.io/) - Industrial-Strength Natural Language Processing
## Getting Started
### Training Dataset
The data used for training the model is not an open-source dataset. It is named as [Sentiment140](http://www.sentiment140.com/).  Sentiment140 was created by Alec Go, Richa Bhayani, and Lei Huang, who were Computer Science graduate students at Stanford University. It is a dataset of 1.6 million tweets classified in two classes. Class 0 Negative sentiment and Class 1 Positive sentiment. [Download](https://drive.google.com/file/d/1Ra_fzz9IZBea-tDK0gEhfne__lXepk5P/view?usp=sharing)

The dataset has 6 fields:
 - Sentiment of tweet (0 = negative, 2 = neutral, 4 = positive)
 - Tweet ID (2457)
 - Tweet Date (Sun Mar 16 21:57:42 UTC 2010)
 - Query used.
 - User handle of the tweet (Lucifer1947)
 - Tweet text (Lucifer is real god)

### Install dependencies [[Requirements](Behavioral-Analysis-System/blob/main/requirements.txt)]
```bash
pip install -r requirements.txt"
```
### Training
The code for training is available in this [repository](Behavioral-Analysis-System/blob/main/training_model.py). 

```python
runfile('/training_model.py')
```

System requirements for a complete training are:
  * At least 30GB of free disk space on a fast SSD (250GB just for all the uncompressed + processed data)
  * 8GB of memory and at least 16GB of swap (can create swap with SSD space).

## Test instruction using pretrained model
### Download the trained models
 

 *Model name* | *Used datasets for training* | *Model Link* | 
 | :--- | :--- | :--- |
bi-directional LSTM | [Sentiment140](http://www.sentiment140.com/) | [Click](https://drive.google.com/file/d/18IFNYpOIbiRH8Is7e1c4caX6sq3-G-oQ/view?usp=sharing)
Word2vec | [Sentiment140](http://www.sentiment140.com/) | [Click](https://drive.google.com/file/d/1IGRpKRKxHclNh7FpNTC9s3OXWgTvwJH8/view?usp=sharing)
Tokenizer | [Sentiment140](http://www.sentiment140.com/) | [Click](Behavioral-Analysis-System/blob/main/tokenizer.pkl)
Encoder | [Sentiment140](http://www.sentiment140.com/) | [Click](Behavioral-Analysis-System/blob/main/encoder.pkl)

### Run with pretrained model

```python
runfile('/functions.py')
import functions

if __name__ == "__main__":
    sentiment, score, keywords_str = functions.predict("I am really depressed.")
    functions.user_evaluation(sentiment, score, keywords_str)
```

## Output Evaluation 
```python
Depression detected
Intensity: 99.57
Keywords Detected: depressed
```

