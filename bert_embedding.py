from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sentence_transformer_model = SentenceTransformer("all-distilroberta-v1")  # todo: also try ('all-MiniLM-L6-v2'), legal-bert
sentence_transformer_model_mpnet = SentenceTransformer("all-mpnet-base-v2")
sentence_transformer_model_legal_bert = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

def get_text_embedding_mpnet(text: str):
    return sentence_transformer_model_mpnet.encode(text, show_progress_bar=False)

def get_text_embedding_legal_bert(text: str):
    return sentence_transformer_model_legal_bert.encode(text, show_progress_bar=False)

def get_text_embedding(text: str):
    # todo: also try summarization before BERT
    return sentence_transformer_model.encode(text, show_progress_bar=False)

# example_texts = ["today is gonna be a great day", "i dont feel so good today",
#                  "you know it rains when there are such dark clouds in the sky",
#                  "i hate this weather", "i love the sun", "there is nothing better than being here with you",
#                  "i love you",
#                  "i hate you", "have a nice day", "what a lovely day", "lets have fun in the sun",
#                  "dont be so rude", "i cant help you right now", "do you need any help"]
#
