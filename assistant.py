from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
import nltk

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the summarizer pipeline
summarizer = pipeline("summarization")

def generate_summary(story_text):
    summary = summarizer(story_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def get_nouns(story_text):
    tokens = word_tokenize(story_text)
    tagged = nltk.pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    return nouns

def create_mcqs(story_text):
    nouns_in_story = get_nouns(story_text)
    sentences = sent_tokenize(story_text)
    mcq_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        
        for i, (word, tag) in enumerate(tagged_words):
            if tag.startswith('NN') and word in nouns_in_story:
                words[i] = "______"
        
        mcq_sentence = ' '.join(words)
        mcq_sentences.append(mcq_sentence)
    
    return mcq_sentences

# Example usage


# Define and load the T5 model and tokenizer for question generation
model_name_qg = "valhalla/t5-base-qg-hl"
tokenizer_qg = T5Tokenizer.from_pretrained(model_name_qg)
model_qg = T5ForConditionalGeneration.from_pretrained(model_name_qg)

def generate_questions(story_text):
    # Prepare the input text with the appropriate prefix
    input_text = "generate questions: " + story_text
    inputs = tokenizer_qg.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model_qg.generate(inputs, max_length=150, num_beams=5, num_return_sequences=5, early_stopping=True)
    questions = [tokenizer_qg.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

# Define and load the Marian model and tokenizer for translation
model_name_urdu = 'Helsinki-NLP/opus-mt-en-ur'
tokenizer_urdu = MarianTokenizer.from_pretrained(model_name_urdu)
model_urdu = MarianMTModel.from_pretrained(model_name_urdu)

def translate_to_urdu(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    print(f"Original sentences: {sentences}")
    
    translated_sentences = []
    for sentence in sentences:
        inputs = tokenizer_urdu.encode(sentence, return_tensors='pt', max_length=512, truncation=True)
        translated = model_urdu.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        translated_sentence = tokenizer_urdu.decode(translated[0], skip_special_tokens=True)
        
        print(f"Original sentence: {sentence}")
        print(f"Translated sentence: {translated_sentence}")
        
        translated_sentences.append(translated_sentence)
    
    # Join the translated sentences back into a single string
    translated_text = ' '.join(translated_sentences)
    print(f"Final translated text: {translated_text}")
    return translated_text

def generate_story_data(story_text):
    story_text_urdu = translate_to_urdu(story_text)
    summary = generate_summary(story_text)
    questions = generate_questions(story_text)
    mcqs = create_mcqs(story_text)
    
    return story_text_urdu, summary, questions, mcqs
