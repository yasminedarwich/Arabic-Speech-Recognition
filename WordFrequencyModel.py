import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')  # Download the necessary NLTK data if you haven't already
nltk.download('stopwords')  # Download the stopwords data

# Define the folder containing your text files
folder_path = '/content/drive/MyDrive/Creators_datasets/AllAudioTextData/AllArabicTranscription'

# Create a list to store words from all text files
all_words = []

# Initialize the NLTK word tokenizer
tokenizer = nltk.RegexpTokenizer(r'\w+')

# Get the list of Arabic stopwords
stop_words = set(stopwords.words('arabic'))

# Iterate through each text file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Tokenize the text into words
            words = tokenizer.tokenize(text.lower())  # Convert to lowercase
            # Filter out stopwords
            words = [word for word in words if word not in stop_words]
            all_words.extend(words)

# Count the frequency of each word
word_counter = Counter(all_words)

# Get the top 30 most frequent words
top_30_words = word_counter.most_common(30)

# Print the top 30 words and their frequencies
for word, count in top_30_words:
    print(f'{word}: {count}')

