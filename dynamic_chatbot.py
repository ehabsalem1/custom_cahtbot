# dynamic_pdf_chatbot.py

import PyPDF2
import difflib

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_into_sentences(text):
    # Very simple split by dot, you can improve with nltk.sent_tokenize
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def find_best_answer(question, sentences):
    # Find sentence that matches closest with user question using SequenceMatcher
    question = question.lower()
    best_ratio = 0
    best_sentence = "Sorry, I couldn't find an answer to that."
    for sentence in sentences:
        ratio = difflib.SequenceMatcher(None, question, sentence.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_sentence = sentence
    return best_sentence

def main():
    pdf_path = input("Enter PDF file path: ")
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    print("You can now ask questions related to the PDF content. Type 'exit' to quit.")
    
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        answer = find_best_answer(question, sentences)
        print("Bot:", answer)

if __name__ == "__main__":
    main()
