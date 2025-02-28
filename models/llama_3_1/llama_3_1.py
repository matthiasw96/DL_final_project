from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login



class llama_3_1:
    def __init__(self, database_path):
        login()
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.database = FAISS.load_local(database_path, self.embedding_model, allow_dangerous_deserialization=True)

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

        self.k = 5


    def invoke(self, question):
        context = self.get_context(question)
        return self.get_answer(question, context)

    def get_context(self, question):
        contexts = self.database.similarity_search(question, self.k)
        context = ""
        for doc in contexts:
            context = context + doc.page_content + "\n"
        return context

    def get_answer(self, question, contexts):
        messages = self.create_messages(question, contexts)

    def create_messages(self, question, contexts):
        pass
