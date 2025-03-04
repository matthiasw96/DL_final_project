from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class deepseek_r1_distill_llama_8B:
    def __init__(self, params):
        self.chunk_size = int(params[0])
        self.k_articles = int(params[1])
        self.k_chunks = int(params[2])

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
        messages = self.create_messages(question, context)
        raw_output = self.generate_answer(messages)
        answer = self.extract_answer(raw_output)
        return answer

    def create_messages(self, question, contexts):
        instruction = """Answer the question in one sentence or less. Do not explain or elaborate. Only provide the direct answer. Mark your answer with the word "Answer:".

        Here is an example:

        Context: France is a country in Europe. Its capital is Paris.
        Question: What is the capital of France?
        Answer: Paris

        Now answer the following question using the provided context.
        """
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Context: {contexts}\n\nQuestion: {question}"}
        ]
        return messages

    def generate_answer(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=600,
            temperature=0.1,
            top_p=0.2,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def extract_answer(self, raw_output):
        answer = raw_output.split("Final Answer:")[-1]
        if " " in answer:
            answer = answer.replace(" ", "").strip()
        return answer