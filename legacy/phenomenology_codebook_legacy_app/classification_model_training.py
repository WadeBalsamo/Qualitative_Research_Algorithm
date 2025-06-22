"""This is test code to train a model based on queried arXiv papers, 

Can be used to evaluate a drafted paper as test_data.txt 

Also provide categories to use next sentence prediction to classify primary source qualitative text data """

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import random

# Constants
vocab_size = 10000  # Assumed vocabulary size
max_seq_length = 100  # Maximum sequence length
d_model = 512  # Dimension of model
num_heads = 8  # Number of heads in multi-head attention
num_layers = 6  # Number of transformer layers
dropout_rate = 0.1  # Dropout rate
batch_size = 32  # Batch size for training

# Custom Dataset for handling Next Sentence Prediction (NSP) task
class NSPDataset(Dataset):
    """
    A custom dataset class for Next Sentence Prediction (NSP).
    This class handles the reading of a text file, splitting it into sentences,
    and preparing positive (consecutive sentences) and negative (random sentence pairs) examples.
    """
    def __init__(self, filename, tokenizer):
        self.sentence_pairs = []
        self.labels = []
        self.tokenizer = tokenizer
        
        # Read the file and prepare NSP dataset
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            for i in range(len(lines) - 1):
                # Positive example
                self.sentence_pairs.append((lines[i], lines[i + 1]))
                self.labels.append(1)

                # Negative example (random sentence)
                random_line = random.choice(lines)
                self.sentence_pairs.append((lines[i], random_line))
                self.labels.append(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent_a, sent_b = self.sentence_pairs[idx]
        label = self.labels[idx]

        # Tokenize and prepare inputs as per the model's requirement
        input_ids = self.tokenizer.encode(sent_a, sent_b, add_special_tokens=True)
        
        # Pad or truncate input_ids to max_seq_length
        input_ids = input_ids[:max_seq_length] + [0] * (max_seq_length - len(input_ids))

        return torch.tensor(input_ids), torch.tensor(label)

# Define the Transformer Model for NSP
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout_rate):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_seq_length, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, dropout_rate) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.nsp_classifier = nn.Linear(d_model, 2)  # Output two logits for binary classification

    def forward(self, x, mask=None):
        x = self.embedding(x) + self.positional_encoding[:x.size(1)]
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.linear(x)
        # Extract [CLS] token's output for NSP task (assuming [CLS] is at position 0)
        cls_output = logits[:, 0, :]
        nsp_logits = self.nsp_classifier(cls_output)
        return nsp_logits    
    
    def _generate_positional_encoding(self, max_len, d_model):
        # Generate positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

# Define the Transformer Layer
class TransformerLayer(nn.Module):
    """
    A single layer of the Transformer model consisting of Multi-Head Attention and Feed Forward layers.
    """

    def __init__(self, d_model, num_heads, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask=None):
        attn_output = self.multi_head_attention(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout2(ff_output))
        return x

# Define Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism, allowing the model to jointly attend to information from different representation subspaces.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            matmul_qk += (mask * -1e9)
        attn_weights = nn.functional.softmax(matmul_qk, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# Define Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Training and Evaluation Functions
def train_model(model, criterion, optimizer, data_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, -1)
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

# Instantiate the model, loss function, and optimizer
model = Transformer(vocab_size, d_model, num_heads, num_layers, dropout_rate)
nsp_criterion = nn.CrossEntropyLoss()  # Loss function for NSP
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for NSP
def train_nsp_model(model, criterion, optimizer, data_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_ids, labels in data_loader:
            optimizer.zero_grad()
            nsp_logits = model(input_ids)
            loss = criterion(nsp_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation function for NSP
def evaluate_nsp_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, labels in data_loader:
            nsp_logits = model(input_ids)
            _, predicted = torch.max(nsp_logits.data, -1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Function to compare sentences and find the most likely continuation of a paragraph
def classify_text_categories(text, categories, tokenizer, model):
    scores = []
    
    for category in categories:
        # Tokenize the text and the category as a pair
        input_ids = tokenizer.encode(text, category, add_special_tokens=True)
        
        # Pad or truncate input_ids to max_seq_length
        input_ids = input_ids[:max_seq_length] + [0] * (max_seq_length - len(input_ids))
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
        
        # Forward pass through the model
        with torch.no_grad():
            nsp_logits = model(input_ids)
        
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(nsp_logits, dim=-1)
        
        # The probability that category is a continuation of the text is the second element
        continuation_probability = probabilities[0][1].item()
        scores.append(continuation_probability)
    
    # Return the category with the highest probability of being a continuation
    max_index = scores.index(max(scores))
    return categories[max_index], scores





##Scrape the training data

import requests
import time
import xml.etree.ElementTree as ET
from urllib.request import urlopen, HTTPError
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

ARXIV = "{http://arxiv.org/OAI/arXiv/}"
OAI = "{http://www.openarchives.org/OAI/2.0/}"
BASE = "http://export.arxiv.org/api/query?"

class LitReviewScraper:
    """
    A class to hold info about attributes of scraping,
    such as date range, categories, and number of returned
    records. The class scrapes papers based on a given query
    and downloads their information, including full texts.
    """
    def __init__(self, xml_record):
        self.xml = xml_record
        # Extracts various pieces of information from the XML
        self.id = self._get_text(ARXIV, "id")
        self.url = "https://arxiv.org/abs/" + self.id
        self.title = self._get_text(ARXIV, "title")
        self.abstract = self._get_text(ARXIV, "abstract")
        self.cats = self._get_text(ARXIV, "categories")
        self.created = self._get_text(ARXIV, "created")
        self.updated = self._get_text(ARXIV, "updated")
        self.doi = self._get_text(ARXIV, "doi")
        self.authors = self._get_authors()
        self.affiliation = self._get_affiliation()
        self.full_text = self._fetch_full_text()

    def _get_text(self, namespace, tag):
        """Extracts text from an XML field"""
        try:
            return self.xml.find(namespace + tag).text.strip().lower().replace("\n", " ")
        except AttributeError:
            return ""

    def _get_name(self, parent, attribute):
        """Extracts author name from an XML field"""
        try:
            return parent.find(ARXIV + attribute).text.lower()
        except AttributeError:
            return "n/a"

    def _get_authors(self):
        """Extract name of authors"""
        authors_xml = self.xml.findall(ARXIV + "authors/" + ARXIV + "author")
        last_names = [self._get_name(author, "keyname") for author in authors_xml]
        first_names = [self._get_name(author, "forenames") for author in authors_xml]
        full_names = [a + " " + b for a, b in zip(first_names, last_names)]
        return full_names

    def _get_affiliation(self):
        """Extract affiliation of authors"""
        authors = self.xml.findall(ARXIV + "authors/" + ARXIV + "author")
        try:
            affiliation = [author.find(ARXIV + "affiliation").text.lower() for author in authors]
            return affiliation
        except AttributeError:
            return []




    def _fetch_full_text(self):
        """
        Fetches full text of the paper from arXiv, trying multiple methods.
        """
        # Phase 1: Try to get full text from the PDF
        full_text = self._fetch_from_pdf(self.url.replace("abs", "pdf"))
        if full_text:
            return full_text
        
        # Phase 2: If PDF parsing fails, try to scrape from the HTML page
        full_text = self._scrape_from_html(self.url)
        if full_text:
            return full_text

        # Phase 3: If all else fails, log an error and return an empty string
        print(f"Failed to fetch full text for {self.url}")
        return ""

    def _fetch_from_pdf(self, pdf_url):
        """
        Fetches text from a PDF URL using PyMuPDF.
        """
        try:
            # Download PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            # Open the PDF
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except Exception as e:
            print(f"Error fetching from PDF for {pdf_url}: {str(e)}")
            return None

    def _scrape_from_html(self, html_url):
        """
        Tries to scrape full text from the HTML page of the paper.
        """
        try:
            response = requests.get(html_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Depending on the structure of the HTML page, you might need a different selector
            full_text_section = soup.find('div', id='full-text')  
            if full_text_section:
                return full_text_section.get_text(separator="\n", strip=True)
            else:
                print(f"Full text section not found in HTML for {html_url}")
                return None
        except Exception as e:
            print(f"Error scraping HTML for {html_url}: {str(e)}")
            return None


    def output(self):
        """Data for each paper record, including full text"""
        return {
            "title": self.title,
            "id": self.id,
            "abstract": self.abstract,
            "categories": self.cats,
            "doi": self.doi,
            "created": self.created,
            "updated": self.updated,
            "authors": self.authors,
            "affiliation": self.affiliation,
            "url": self.url,
            "full_text": self.full_text
        }

class Scraper:
    """
    Scrapes data from arXiv based on given parameters like category, date range, and filters.
    """

    def __init__(self, category: str, max_results: int = 10000):
        self.cat = str(category)
        self.max_results = max_results
        # Set up the base URL with the category and date range
        self.url = BASE + "search_query=%s" % self.cat
        # Set up a counter to keep track of the number of fetched records
        self.fetched_records = 0

    def scrape(self):
        """Performs the scraping operation."""
        t0 = time.time()
        url = self.url
        records = []
        while True and self.fetched_records < self.max_results:
            try:
                response = urlopen(url)
                xml = response.read()
                root = ET.fromstring(xml)
                for record_xml in root.findall(OAI + "ListRecords/" + OAI + "record"):
                    meta = record_xml.find(OAI + "metadata").find(ARXIV + "arXiv")
                    record = LitReviewScraper(meta).output()
                    records.append(record)
                    self.fetched_records += 1
                    if self.fetched_records >= self.max_results:
                        break

                # Handle pagination with resumptionToken
                token = root.find(OAI + "ListRecords").find(OAI + "resumptionToken")
                if token is None or token.text is None or self.fetched_records >= self.max_results:
                    break
                else:
                    url = BASE + "resumptionToken=%s" % token.text

            except HTTPError as e:
                if e.code == 503:
                    # Handle rate limiting
                    time.sleep(10)
                    continue
                else:
                    raise

        t1 = time.time()
        print(f"Scraping completed in {t1 - t0:.1f} seconds.")
        print(f"Total number of records: {len(records)}")
        return records

def classify_data_with_litreview(query, categories=[], original_research_qualitative_data_path=False, test_data_path=False):

    # Step 1: Data Collection with arXiv Scraper
    scraper = LitReviewScraper(query=query)
    training_data = scraper.scrape()
    with open('training_data.txt', 'w') as file:
        file.write(training_data)

    # Load the training and testing data
    # Assuming you have a tokenizer and NSPDataset class
    train_data = NSPDataset('training_data.txt', tokenizer)
    test_data = NSPDataset(test_data_path, tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if test_data:
      test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Initialize the Model, Loss Function, and Optimizer
    model = Transformer(vocab_size, d_model, num_heads, num_layers, dropout_rate)
    criterion = nn.CrossEntropyLoss()  # Loss function for NSP
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_nsp_model(model, criterion, optimizer, train_loader, num_epochs=10)
    if test_data:
      # Evaluate the model
      test_accuracy = evaluate_nsp_model(model, test_loader)
      print(f'Test Accuracy: {test_accuracy:.2f}')
    if original_research_qualitative_data_path and categories:
      # Step 5: Classify Original Research Qualitative Data
      with open(original_research_qualitative_data_path, 'r') as file:
          texts_to_classify = file.readlines()
      
      # Classify text categories
      best_categories = [classify_text_categories(text, categories, tokenizer, model) for text in texts_to_classify]
      classification_score = {category: best_categories.count(category) for category in categories}
      
      # Print or save the classified sentences and their scores
      for text, category in zip(texts_to_classify, best_categories):
          print(f"Text: {text}, Classified as: {category}")
      print(f"Classification Score: {classification_score}")

if __name__ == "__main__":
    classify_data_with_litreview(
        query="chronic pain",
        test_data_path='drafted_paper.txt',
        original_research_qualitative_data_path='research_data.txt',
        categories=[
            'this means the therapy is effective or positive effects', 
            'this means the therapy is negative or harmful side effects', 
            'this doesn\'t relate to the therapeutic efficacy'
        ]
    )
