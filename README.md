# Emotion Classifier

A beginner-friendly Machine Learning project that classifies text into three sentiment categories:

* **Positive**
* **Negative**
* **Neutral**

The project uses the **TweetEval Sentiment Dataset** from Hugging Face and a traditional Machine Learning approach using **TF-IDF** and **Logistic Regression**.

The trained model is integrated into a **Streamlit** web application where users can enter text and receive a sentiment prediction with a confidence score.

---

## Project Demo

The application allows users to enter text such as:

> I am extremely happy today!

The model returns:

```text
Prediction: Positive
Confidence: XX.XX%
```

---

## Features

* Text sentiment classification
* Three sentiment classes:

  * Positive
  * Negative
  * Neutral
* TweetEval sentiment dataset
* Text preprocessing
* TF-IDF feature extraction
* Logistic Regression classifier
* Accuracy evaluation
* Precision evaluation
* Recall evaluation
* F1-score evaluation
* Confusion matrix
* Prediction confidence/probability
* Custom text testing
* Streamlit web interface
* Trained model saved using Joblib
* Ready for deployment with Streamlit Community Cloud

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Hugging Face Datasets**
* **Joblib**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Google Colab**
* **GitHub**

---

## Dataset

This project uses the **TweetEval Sentiment Dataset** from Hugging Face.

TweetEval is a benchmark containing several Twitter-related NLP tasks. For this project, the **Sentiment** task is used because it contains the three classes required by this application:

| Label | Sentiment |
| ----- | --------- |
| 0     | Negative  |
| 1     | Neutral   |
| 2     | Positive  |

The dataset provides separate:

* Training set
* Validation set
* Test set

Dataset:

**TweetEval Sentiment**

Hugging Face:

https://huggingface.co/datasets/cardiffnlp/tweet_eval

---

## Machine Learning Approach

The project uses the following pipeline:

```text
Input Text
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Sentiment Prediction
    ↓
Positive / Neutral / Negative
```

### 1. Text Preprocessing

The initial preprocessing step converts text to lowercase.

Example:

```text
"I AM VERY HAPPY!"
```

becomes:

```text
"i am very happy!"
```

The preprocessing is intentionally kept simple because this project is designed to demonstrate the fundamentals of Natural Language Processing and Machine Learning.

---

### 2. TF-IDF

TF-IDF stands for:

**Term Frequency - Inverse Document Frequency**

Machine-learning algorithms cannot directly work with raw text. TF-IDF converts text into numerical feature vectors.

The project uses:

```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)
```

This allows the model to consider both individual words and two-word combinations.

For example:

```text
happy
very happy
not good
```

---

### 3. Logistic Regression

The numerical TF-IDF features are given to a Logistic Regression classifier.

The model learns relationships between text features and sentiment labels.

The final prediction is one of:

```text
Positive
Neutral
Negative
```

The model also supports probability estimates using:

```python
model.predict_proba()
```

---

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The evaluation is performed using the TweetEval test set, which was not used to train the model.

### Results

Replace the values below with the actual results from your Google Colab notebook:

```text
Accuracy: XX.XX%
Precision: XX.XX%
Recall: XX.XX%
F1-score: XX.XX%
```

---

## Confusion Matrix

The confusion matrix is used to understand which sentiment classes the model predicts correctly and which classes it confuses.

The three classes are:

```text
Negative
Neutral
Positive
```

Add your confusion matrix image here if you want:

```markdown
![Confusion Matrix](images/confusion_matrix.png)
```

---

## Project Structure

```text
emotion-classifier/
│
├── app.py
│
├── model/
│   ├── emotion_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── emotion_classifier.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

#### `app.py`

The Streamlit application.

It:

* Accepts user input
* Loads the trained model
* Converts the input into TF-IDF features
* Predicts sentiment
* Displays the prediction
* Displays prediction confidence

#### `model/emotion_model.pkl`

The trained Logistic Regression model.

#### `model/tfidf_vectorizer.pkl`

The trained TF-IDF vectorizer used to convert text into numerical features.

#### `notebooks/emotion_classifier.ipynb`

The Google Colab/Jupyter Notebook containing:

* Dataset loading
* Data exploration
* Preprocessing
* TF-IDF
* Model training
* Evaluation
* Custom testing
* Model saving

#### `requirements.txt`

Contains the Python dependencies required to run the Streamlit application.

#### `.gitignore`

Contains files and folders that should not be uploaded to GitHub.

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd emotion-classifier
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually similar to:

```text
http://localhost:8501
```

Open that address in your browser.

---

## Example

### Input

```text
I am extremely happy today!
```

### Output

```text
Prediction: Positive
```

The application also displays the model's estimated confidence.

---

## Custom Testing

You can test different types of sentences.

### Positive

```text
I absolutely love this new game!
```

### Negative

```text
This was a terrible experience.
```

### Neutral

```text
I went to the store today.
```

The model will classify each input as:

```text
Positive
Negative
Neutral
```

---

## Important Limitation

This project uses **TF-IDF + Logistic Regression**, which is a traditional Machine Learning approach.

The model learns statistical relationships between words/phrases and sentiment. It does not understand language in the same way that modern transformer-based models do.

For example, sentences involving negation or sarcasm can sometimes be difficult:

```text
I am not happy.
```

or:

```text
Yeah, that's just great...
```

Therefore, the prediction should not be treated as perfect or as a human-level understanding of emotion.

---

## Future Improvements

Possible improvements for future versions include:

* Use a transformer model such as BERT
* Fine-tune DistilBERT
* Experiment with BERTweet
* Improve text preprocessing
* Handle negation more effectively
* Detect sarcasm
* Add probability charts
* Add batch prediction
* Allow CSV file uploads
* Compare multiple Machine Learning models
* Add model performance visualizations
* Improve the Streamlit user interface

---

## Learning Objectives

This project was created to understand the complete Machine Learning workflow:

```text
Dataset
   ↓
Data Exploration
   ↓
Preprocessing
   ↓
Feature Extraction
   ↓
Model Training
   ↓
Validation
   ↓
Testing
   ↓
Evaluation
   ↓
Model Saving
   ↓
Application Development
   ↓
Deployment
```

The project is intended as a beginner-friendly introduction to **Natural Language Processing (NLP)** and **Machine Learning classification**.

---

## Deployment

The Streamlit application can be deployed using **Streamlit Community Cloud**.

Basic deployment steps:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Select the `main` branch.
6. Select `app.py` as the application file.
7. Deploy the application.

---

## License

This project is intended for educational purposes.

Add an appropriate license if you decide to distribute or reuse the project publicly.

---

## Author

**Muhammad Bilal**

Computer Science Student | Aspiring Game Developer | AI/ML Learner

---

## Acknowledgements

* Hugging Face for providing the TweetEval dataset.
* Cardiff NLP for the TweetEval benchmark.
* Scikit-learn for the Machine Learning tools.
* Streamlit for the web application framework.
