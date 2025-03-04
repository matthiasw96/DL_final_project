from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class llama_3_1:
    def __init__(self, params):
        self.chunk_size = params[0]
        self.k_articles = params[1]
        self.k_chunks = params[2]

        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.database = FAISS.load_local("database", self.embedding_model, allow_dangerous_deserialization=True)
        self.device = "cuda:0"

        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.initialize_model(model_name=self.model_name)

        self.text_splitter = self.create_text_splitter()

    def initialize_model(self, model_name):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        return model

    def create_text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            length_function=len,
        )
        return text_splitter

    def invoke(self, question):
        context = self.get_context(question)
        return self.get_answer(question, context)

    def get_context(self, question):
        articles = self.database.similarity_search(question, self.k_articles)
        chunks = self.split_articles(articles)
        documents = self.filter_chunks(chunks, question)
        context = ""
        for doc in documents:
            context = context + doc.page_content + "\n\n"
        return context

    def split_articles(self, articles):
        chunks = []
        doc_chunks = []
        last_chunk = ""

        for article in articles:
            article_chunks = self.text_splitter.split_text(article.page_content)
            chunks.extend(article_chunks)

        for chunk in chunks:
            if len(chunk) < self.chunk_size / 3:
                last_chunk += chunk
            else:
                doc_chunks.append(last_chunk)
                last_chunk = chunk
        doc_chunks.append(last_chunk)

        return doc_chunks

    def filter_chunks(self, chunks, question):
        lib = FAISS.from_texts(chunks, self.embedding_model)
        top_chunks = lib.similarity_search(question, self.k_chunks)
        return top_chunks

    def get_answer(self, question, context):
        message = self.create_message(question, context)
        raw_output = self.generate_answer(message)
        answer = self.extract_answer(raw_output)
        return answer

    def create_message(self, question, contexts):
        message = f"""<|begin_of_text|>
          <|start_header_id|>system<|end_header_id|>
          You are an AI assistant that always answers as quickly as possible. 
          Use the following context to answer the question.

          <|start_header_id|>user<|end_header_id|>
          Question: {question}

          Context: {contexts}

          <|start_header_id|>assistant<|end_header_id|>"""
        return message

    def generate_answer(self, message):
        inputs = self.tokenizer(message, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return raw_output

    def extract_answer(self, raw_output):
      answer_start = raw_output.split("assistant\n")[1].strip()
      answer = answer_start[:answer_start.index("\n")]
      return answer