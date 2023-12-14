import os
import nltk
import re
import networkx as nx
from flask import Flask, render_template, request
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    pipeline,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)
from evaluate import load
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models.ldamodel import LdaModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Initialize summarization models and ROUGE metric
t5_model_name = "t5-small"
t5_summarizer = pipeline("summarization", model=t5_model_name)
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_summarizer = BartForConditionalGeneration.from_pretrained(bart_model_name)
pegasus_model_name = "google/pegasus-cnn_dailymail"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_summarizer = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
rouge_metric = load("rouge")

# Functions for summarization and keyword extraction
def slider_to_length(slider_value):
    return 30 * slider_value

def calculate_rouge_scores(original_text, summarized_text):
    if not original_text or not summarized_text:
        return {}
    scores = rouge_metric.compute(predictions=[summarized_text], references=[original_text])
    for score_type in scores:
        scores[score_type] = round(scores[score_type], 4)
    scores_str = str(scores).replace("{", "").replace("}", "")
    return scores_str

def summarize_with_t5(input_text, length, slider_value):
    if not input_text:
        return "", []
    summary = t5_summarizer(input_text, max_length=length, min_length=length - 5, do_sample=False)
    full_summarized_text = summary[0]["summary_text"]
    sentences = sent_tokenize(full_summarized_text)
    summarized = " ".join(sentences[:slider_value]).replace(" .", ".")
    dictionary, corpus = create_lda_corpus(summarized)
    lda_model = train_lda_model(dictionary, corpus)
    t5_keywords = extract_keywords(lda_model)
    return summarized, t5_keywords

def summarize_with_bart(input_text, length, slider_value):
    if not input_text:
        return "", []
    input_ids = bart_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary = bart_summarizer.generate(input_ids, max_length=length, min_length=length - 5)
    full_summarized_text = bart_tokenizer.decode(summary[0], skip_special_tokens=True)
    sentences = sent_tokenize(full_summarized_text)
    summarized = " ".join(sentences[:slider_value])
    dictionary, corpus = create_lda_corpus(summarized)
    lda_model = train_lda_model(dictionary, corpus)
    bart_keywords = extract_keywords(lda_model)
    return summarized, bart_keywords

def summarize_with_pegasus(input_text, length, slider_value):
    if not input_text:
        return "", []
    input_ids = pegasus_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary = pegasus_summarizer.generate(input_ids, max_length=length, min_length=length - 5)
    full_summarized_text = pegasus_tokenizer.decode(summary[0], skip_special_tokens=True)
    sentences = sent_tokenize(full_summarized_text)
    summarized = " ".join(sentences[:slider_value]).replace("<n>", "")
    dictionary, corpus = create_lda_corpus(summarized)
    lda_model = train_lda_model(dictionary, corpus)
    pegasus_keywords = extract_keywords(lda_model)
    return summarized, pegasus_keywords

def summarize_with_textrank(input_text, slider_value):
    if not input_text.strip():
        return "", []
    sentences = sent_tokenize(input_text)
    stop_words = stopwords.words("english")
    sentences_clean = [
        " ".join(
            re.sub(r"[^\w\s]", "", word).lower()
            for word in sentence.split()
            if word.lower() not in stop_words
        )
        for sentence in sentences
    ]
    tfidf_vectorizer = TfidfVectorizer()
    sentence_vectors = tfidf_vectorizer.fit_transform(sentences_clean).toarray()
    similarity_matrix = cosine_similarity(sentence_vectors)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [sentence for _, sentence in ranked_sentences[:slider_value]]
    summarized = " ".join(top_sentences)
    dictionary, corpus = create_lda_corpus(summarized)
    lda_model = train_lda_model(dictionary, corpus)
    textrank_keywords = extract_keywords(lda_model)
    return summarized, textrank_keywords

def create_lda_corpus(text):
    tokenized_text = [word for word in text.lower().split() if word not in stopwords.words("english")]
    dictionary = corpora.Dictionary([tokenized_text])
    corpus = [dictionary.doc2bow(tokenized_text)]
    return dictionary, corpus

def train_lda_model(dictionary, corpus):
    lda_model = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
    return lda_model

def extract_keywords(lda_model):
    topics = lda_model.print_topics(num_words=5)
    keywords = [word.split("*")[1].strip() for _, words in topics for word in words.split("+")]
    return keywords

# Flask route handler
@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    t5_summarized_text, bart_summarized_text, textrank_summarized_text, pegasus_summarized_text = "", "", "", ""
    t5_rouge_scores, bart_rouge_scores, textrank_rouge_scores, pegasus_rouge_scores = {}, {}, {}, {}
    t5_keywords, bart_keywords, textrank_keywords, pegasus_keywords = [], [], [], []

    if request.method == "POST":
        original_text = request.form["original_text"]
        slider_value = int(request.form["sentence_count"])
        summary_length = slider_to_length(slider_value)
        t5_summarized_text, t5_keywords = summarize_with_t5(original_text, summary_length, slider_value)
        bart_summarized_text, bart_keywords = summarize_with_bart(original_text, summary_length, slider_value)
        pegasus_summarized_text, pegasus_keywords = summarize_with_pegasus(original_text, summary_length, slider_value)
        textrank_summarized_text, textrank_keywords = summarize_with_textrank(original_text, slider_value)
        t5_rouge_scores = calculate_rouge_scores(original_text, t5_summarized_text)
        bart_rouge_scores = calculate_rouge_scores(original_text, bart_summarized_text)
        textrank_rouge_scores = calculate_rouge_scores(original_text, textrank_summarized_text)
        pegasus_rouge_scores = calculate_rouge_scores(original_text, pegasus_summarized_text)

    return render_template(
        "index.html",
        original_text=original_text,
        t5_summarized_text=t5_summarized_text,
        bart_summarized_text=bart_summarized_text,
        textrank_summarized_text=textrank_summarized_text,
        pegasus_summarized_text=pegasus_summarized_text,
        t5_rouge_scores=t5_rouge_scores,
        bart_rouge_scores=bart_rouge_scores,
        textrank_rouge_scores=textrank_rouge_scores,
        pegasus_rouge_scores=pegasus_rouge_scores,
        t5_keywords=t5_keywords,
        bart_keywords=bart_keywords,
        textrank_keywords=textrank_keywords,
        pegasus_keywords=pegasus_keywords,
    )

if __name__ == "__main__":
    app.run(debug=True, port=5005)
