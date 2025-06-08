from langchain.embeddings.base import Embeddings

class SBERTWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts):
        return [self.model.encode(t, convert_to_numpy=True).tolist() for t in texts]