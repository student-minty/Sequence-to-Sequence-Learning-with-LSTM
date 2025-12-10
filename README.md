#📘 Seq2Seq LSTM Machine Translation – English → French Translator

A Deep Learning–based Encoder–Decoder implementation using LSTMs

⭐ Project Overview

This project implements a Sequence-to-Sequence (Seq2Seq) neural network using LSTM layers to translate English sentences into French. The model is trained on an English–French sentence dataset and follows an encoder–decoder architecture commonly used in machine translation tasks.

🚀 Key Features

Encoder–decoder architecture using LSTM networks

Custom preprocessing pipeline:

tokenization

vocabulary creation

word-to-index mapping

padding & truncation

Trained on 10,000 English–French sentence pairs

Evaluates translation quality using:

BLEU Score

Sample qualitative outputs

Fully implemented in TensorFlow / Keras

📂 Dataset

Source: https://www.manythings.org/anki/fra-eng.zip

Contains English–French sentence pairs (fra.txt)

Only the last 10,000 pairs are used for faster training

Dataset structure:

English_sentence \t French_sentence

🧹 Data Preprocessing Pipeline

Load the dataset

Clean text (lowercase, remove punctuation)

Tokenize English & French separately

Build vocabularies

Convert text → integer sequences

Pad sequences to a fixed length

Train-test split (80/20)

🧠 Model Architecture
Encoder

Embedding Layer

LSTM Layer (units = 128 / 256 / 512 depending on experiment)

Outputs encoder hidden & cell states

Decoder

Embedding Layer

LSTM Layer receiving encoder state

Dense Layer with Softmax activation

⚙️ Training Configuration

Loss: sparse_categorical_crossentropy

Optimizer: Adam

Metrics: Accuracy

Epochs: configurable

Batch size: configurable

📊 Evaluation
Quantitative

BLEU Score using nltk.translate.bleu_score

Evaluation on test split (20%)

Qualitative

Sample input English sentence

Model translation output (French)

Comparison with ground-truth translation

🔬 Experiments

You vary the number of LSTM units to compare performance:

128 units

256 units

512 units

You also discuss how sequence length affects:

training stability

translation quality

inference difficulty

📁 Project Structure
Seq2Seq-LSTM-Translation/
│
├── Seq2Seq LSTM.ipynb     # Full implementation notebook
├── README.md              # Project documentation
├── data/                  # (Optional) Dataset files
└── results/               # BLEU scores, sample outputs

🛠️ Technologies Used

Python

NumPy

TensorFlow / Keras

NLTK

Matplotlib

📝 How to Run

Download the dataset:

https://www.manythings.org/anki/fra-eng.zip


Extract fra.txt to the project folder.

Run the Jupyter Notebook:

jupyter notebook "Seq2Seq LSTM.ipynb"


Train the model and view results.

📌 Future Improvements

Beam Search decoding

Attention mechanism (Luong or Bahdanau)

Transformer-based model

Support for larger datasets

👨‍💻 Author

Janvi Kumari
CS564 – Machine Learning | IIT Patna
