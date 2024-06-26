# install all the dependencies 

# %pip install beautifulsoup4
# %pip install selenium
# %pip install webdriver-manager
# %pip install textblob
# %pip install requests
# %pip install pandas nltk textstat openpyxl

# Or you can use ipynb file as I also attached that too just run cells in file


# importing libraries  

from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService 
from webdriver_manager.chrome import ChromeDriverManager
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
import pandas as pd
import textstat
import textstat
import re
import os

# Load the input file
urldata = pd.read_csv('./urls.csv')
urldata = urldata.dropna()
urls = urldata['URL']
urldata

# function to  fetching the web page data
def getpages(urls : list[str]) -> list[str]:
    options = webdriver.ChromeOptions() 
    options.headless = True
    driver = webdriver.Chrome(service=ChromeService( 
        ChromeDriverManager().install()), options=options) 
    pages = []
    for i, each in enumerate(urls):
        print(f'\rFetching page {i+1}/{len(urls)}...', end='')
        driver.get(each)
        pages.append(driver.page_source)
    return pages

pages = getpages(urls)


def getstring(node) -> str:
    children = list(node.children)
    retter = ''
    if len(children) > 1:
        for each in children:
            retter += getstring(each)
    else:
        retter = node.string
    return retter



def scrape_page(rawpage: str, elements: list[str] = None) -> str:
    if elements is None:
        elements = ['p', 'li']
    text = []
    soup = BeautifulSoup(rawpage, 'html.parser')
    article = soup.article
    title_tag = soup.find('h1') # Extractring the title of the article
    if not title_tag:
        title_tag = soup.find('h2')  # <h2> as a fallback
    title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
    
    if article is not None:
        for each in elements:
            each_elements = list(filter(lambda x: 'class' not in x.attrs, article.find_all(each)))
            each_texts = list(map(lambda x: '\n'.join(x.strings), each_elements))
            text += each_texts 
    return title+ '\n' +'\n'.join(text)
try:
    os.mkdir('./articles')
except FileExistsError:
    pass

# Assuming pages is a list of raw HTML page strings and urldata is a DataFrame
for i, each in enumerate(pages):
    text = scrape_page(each)
    with open(f'./articles/{urldata.iloc[i, 0]}.txt', 'w', encoding='utf-8') as file:
        file.write(text)


# downloading the necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords from the StopWords folder
stop_words = set()
for stopword_file in os.listdir('StopWords'):
    with open(os.path.join('StopWords', stopword_file), 'r') as file:
        for line in file:
            word = line.strip()
            if word:
                stop_words.add(word.lower())

# Load positive and negative word dictionaries
positive_words = set()
negative_words = set()

with open('MasterDictionary/positive-words.txt', 'r') as file:
    for line in file:
        word = line.strip()
        if word and word.lower() not in stop_words:
            positive_words.add(word.lower())

with open('MasterDictionary/negative-words.txt', 'r') as file:
    for line in file:
        word = line.strip()
        if word and word.lower() not in stop_words:
            negative_words.add(word.lower())


# functions for text analysis
def clean_text(text):
    words = nltk.word_tokenize(text)
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(cleaned_words)

def get_word_count(text):
    words = nltk.word_tokenize(text)
    return len(words)

def get_sentence_count(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

def get_avg_sentence_length(text):
    words = get_word_count(text)
    sentences = get_sentence_count(text)
    return words / sentences if sentences != 0 else 0

def get_complex_word_count(text):
    words = nltk.word_tokenize(text)
    complex_words = [word for word in words if textstat.syllable_count(word) > 2]
    return len(complex_words)

def get_percentage_complex_words(text):
    word_count = get_word_count(text)
    complex_word_count = get_complex_word_count(text)
    return (complex_word_count / word_count) * 100 if word_count != 0 else 0

def get_fog_index(text):
    return textstat.gunning_fog(text)

def get_syllable_per_word(text):
    words = nltk.word_tokenize(text)
    total_syllables = sum(textstat.syllable_count(word) for word in words)
    return total_syllables / len(words) if len(words) != 0 else 0

def get_avg_word_length(text):
    words = nltk.word_tokenize(text)
    total_length = sum(len(word) for word in words)
    return total_length / len(words) if len(words) != 0 else 0

def get_personal_pronouns(text):
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.I)
    return len(pronouns)

def get_polarity_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_subjectivity_score(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity

def get_positive_score(text):
    words = nltk.word_tokenize(text)
    positive_words_count = sum(1 for word in words if word in positive_words)
    return positive_words_count

def get_negative_score(text):
    words = nltk.word_tokenize(text)
    negative_words_count = sum(1 for word in words if word in negative_words)
    return negative_words_count

def get_polarity_score_calculated(positive_score, negative_score):
    return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

def get_subjectivity_score_calculated(positive_score, negative_score, word_count):
    return (positive_score + negative_score) / (word_count + 0.000001)


# Analyze the articles and populate the output dataframe
output_data = []

for index, row in urldata.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    file_path = f'articles/{url_id}.txt'
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                cleaned_text = clean_text(text)
                
            word_count = get_word_count(cleaned_text)
            positive_score = get_positive_score(cleaned_text)
            negative_score = get_negative_score(cleaned_text)
            analysis = {
                'URL_ID': url_id,
                'URL': url,
                'POSITIVE SCORE': positive_score,
                'NEGATIVE SCORE': negative_score,
                'POLARITY SCORE': get_polarity_score_calculated(positive_score, negative_score),
                'SUBJECTIVITY SCORE': get_subjectivity_score_calculated(positive_score, negative_score, word_count),
                'AVG SENTENCE LENGTH': get_avg_sentence_length(cleaned_text),
                'PERCENTAGE OF COMPLEX WORDS': get_percentage_complex_words(cleaned_text),
                'FOG INDEX': get_fog_index(cleaned_text),
                'AVG NUMBER OF WORDS PER SENTENCE': get_avg_sentence_length(cleaned_text),
                'COMPLEX WORD COUNT': get_complex_word_count(cleaned_text),
                'WORD COUNT': word_count,
                'SYLLABLE PER WORD': get_syllable_per_word(cleaned_text),
                'PERSONAL PRONOUNS': get_personal_pronouns(cleaned_text),
                'AVG WORD LENGTH': get_avg_word_length(cleaned_text)
            }
            
            output_data.append(analysis)
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error analyzing {url_id}: {e}")


# Convert output_data to a DataFrame
df_output = pd.DataFrame(output_data)

# Ensure columns are in the correct order as per the output structure
columns_order = [
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
    'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
    'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
]

df_output = df_output[columns_order]

# Save to the output file
output_file = 'Output.xlsx'
df_output.to_excel(output_file, index=False)
print("Text analysis completed successfully.")
