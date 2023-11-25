import os
from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline, PegasusTokenizer, PegasusForConditionalGeneration
from summa import summarizer
from evaluate import load

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

# Initialize the summarization pipeline with a T5 model
t5_model_name = "t5-small"
t5_summarizer = pipeline("summarization", model=t5_model_name)

# Initialize the BART model for summarization
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_summarizer = BartForConditionalGeneration.from_pretrained(bart_model_name)

# Initialize the PEGASUS model for summarization
pegasus_model_name = "google/pegasus-cnn_dailymail"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_summarizer = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)

# Load the rouge metric
rouge_metric = load("rouge")

# Function to map slider value to length
def slider_to_length(slider_value):
    # Example: Convert slider value to summary length. Adjust as needed.
    return 10 + 15 * slider_value

# Define a function to calculate ROUGE scores
def calculate_rouge_scores(original_text, summarized_text):
    if not original_text or not summarized_text:
        return {}
    scores = rouge_metric.compute(predictions=[summarized_text], references=[original_text])
    
    # Round the scores to two decimal places
    for score_type in scores:
        scores[score_type] = round(scores[score_type], 4)

    scores_str = str(scores).replace('{', '').replace('}', '')
    return scores_str

def summarize_with_t5(input_text, length):
    if not input_text:
        return ""
    summary = t5_summarizer(input_text, max_length=length, min_length=length - 5, do_sample=False)
    summarized_text = summary[0]["summary_text"]
    return summarized_text.replace(" .", ".")

def summarize_with_bart(input_text, length):
    if not input_text:
        return ""
    input_ids = bart_tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary = bart_summarizer.generate(input_ids, max_length=length, min_length=length - 5)
    summarized_text = bart_tokenizer.decode(summary[0], skip_special_tokens=True)
    return summarized_text

# Define a function to perform summarization using TextRank
def summarize_with_textrank(input_text, length):
    if not input_text:
        return ""
    # Example mapping, adjust as needed
    word_count = length # Assuming an average sentence length
    return summarizer.summarize(input_text, words=word_count)

# Define a function to perform summarization using PEGASUS
def summarize_with_pegasus(input_text, length):
    if not input_text:
        return ""
    input_ids = pegasus_tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary = pegasus_summarizer.generate(input_ids, max_length=length, min_length=length - 5)
    summarized_text = pegasus_tokenizer.decode(summary[0], skip_special_tokens=True)
    return summarized_text.replace('<n>', '')

@app.route('/', methods=['GET', 'POST'])
def index():
    original_text = ""
    t5_summarized_text = ""
    bart_summarized_text = ""
    textrank_summarized_text = ""
    pegasus_summarized_text = ""
    t5_rouge_scores = {}
    bart_rouge_scores = {}
    textrank_rouge_scores = {}
    pegasus_rouge_scores = {}

    if request.method == 'POST':
        original_text = request.form['original_text']
        slider_value = int(request.form['sentence_count'])
        summary_length = slider_to_length(slider_value)
        t5_summarized_text = summarize_with_t5(original_text, summary_length)
        bart_summarized_text = summarize_with_bart(original_text, summary_length)
        textrank_summarized_text = summarize_with_textrank(original_text, summary_length)
        pegasus_summarized_text = summarize_with_pegasus(original_text, summary_length)

        # Calculate ROUGE scores for each summarization
        t5_rouge_scores = calculate_rouge_scores(original_text, t5_summarized_text)
        bart_rouge_scores = calculate_rouge_scores(original_text, bart_summarized_text)
        textrank_rouge_scores = calculate_rouge_scores(original_text, textrank_summarized_text)
        pegasus_rouge_scores = calculate_rouge_scores(original_text, pegasus_summarized_text)

    return render_template('index.html', original_text=original_text, 
                           t5_summarized_text=t5_summarized_text, 
                           bart_summarized_text=bart_summarized_text, 
                           textrank_summarized_text=textrank_summarized_text, 
                           pegasus_summarized_text=pegasus_summarized_text,
                           t5_rouge_scores=t5_rouge_scores,
                           bart_rouge_scores=bart_rouge_scores,
                           textrank_rouge_scores=textrank_rouge_scores,
                           pegasus_rouge_scores=pegasus_rouge_scores)

if __name__ == '__main__':
    app.run(debug=True, port=5005)
