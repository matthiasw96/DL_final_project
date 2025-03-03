from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch



class deepseek_r1_distill_llama_8B:
    def __init__(self, params):
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.database = FAISS.load_local("database", self.embedding_model, allow_dangerous_deserialization=True)
        self.device = "cuda:0"

        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.initialize_model(model_name=self.model_name)

        self.text_splitter = self.create_text_splitter()

        self.chunk_size = params["chunk_size"]
        self.k_articles = params["k_articles"]
        self.k_chunks = params["k_chunks"]

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
        documents = self.filter_chunks(chunks)
        context = ""
        for doc in documents:
            context = context + doc.page_content + "\n"
        return context

    def split_articles(self, articles):
        chunks = []
        doc_chunks = []
        last_chunk = ""

        for article in articles:
            article_chunks = self.text_splitter.split_text(article)
            chunks.extend(article_chunks)

        for chunk in chunks:
            if len(chunk) < self.chunk_size / 3:
                last_chunk += chunk
            else:
                doc_chunks.append(last_chunk)
                last_chunk = chunk
        doc_chunks.append(last_chunk)

        return doc_chunks

    def filter_chunks(self, chunks):
        lib = FAISS.from_texts(chunks, embedding_model=self.embedding_model)
        top_chunks = lib.similarity_search(chunks, self.k_chunks)
        return top_chunks

    def get_answer(self, question, context):
        messages = self.create_messages(question, context)
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs)
        answer =  self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer[answer.rfind('\n'):]
        return answer

    def create_messages(self, question, contexts):
        question_add = [" To answer the question extract the information from these texts:",
                        "\nAnswer as shortly as possible, no additional information, no punctuation. "]
        instruction = "You are a chatbot who always responds as shortly as possible."

        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": question
                                        + question_add[0]
                                        + contexts
                                        + question_add[1]
             },
        ]

        return messages