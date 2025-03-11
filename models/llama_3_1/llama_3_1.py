from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class llama_3_1:
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
        message = self.create_message(question, context)
        raw_output = self.generate_answer(message)
        answer = self.extract_answer(raw_output)
        return answer

    def create_message(self, question, contexts):
        message = f"""<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            Role: You are an AI assistant that **only answers based on the provided context**. 
            If the answer is not in the context, say "Not in context".

            <|start_header_id|>user<|end_header_id|>
            Task: Answer the question based on the context using **only one expression**.  
            **If the answer is not in the context, reply "Not in context"**.  
            Do NOT answer based on prior knowledge.  

            ### Here are some examples:
            **Example 1 (Correct Usage)**  
            **Question:** Who directed the movie La Dolce Vita?  
            **Context:** La Dolce Vita is a 1960 Italian film directed by Federico Fellini.  
            **Final Answer:** [START]Federico Fellini[END]  

            **Example 2 (Correct Usage)**  
            **Question:** What was Diana Ross's first solo No. 1?  
            **Context:** Diana Ross released her debut solo album in 1970, which contained "Ain’t No Mountain High Enough," her first solo No. 1 hit.  
            **Final Answer:** [START]Ain't No Mountain High Enough[END]  

            **Example 3 (Correct Usage)**  
            **Question:** Tony Lumpkin, Constance Neville, and George Hastings are all characters in which play?  
            **Context:** She Stoops to Conquer is a play by Oliver Goldsmith that includes Tony Lumpkin, Constance Neville, and George Hastings.  
            **Final Answer:** [START]She Stoops to Conquer[END]  

            **Example 4 (Incorrect Context Example - No Answer in Context)**  
            **Question:** What is the capital of Spain?  
            **Context:** Madrid is known for its rich history and culture, but no explicit mention is made about it being the capital.  
            **Final Answer:** [START]Not in context[END]  

            ---

            ### Now answer the following question using the given context only.
            - **Only use one expression, not a full sentence.**  
            - **If the answer is not found, return 'Not in context'.**  
            - **Surround the answer with [START] and [END] to clearly mark it.**

            **Question:** {question}  
            **Context:** {contexts}  

            Final Answer: [START]
            <|start_header_id|>assistant<|end_header_id|>

            """
        return message

    def generate_answer(self, message):
        inputs = self.tokenizer(message, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return raw_output

    def extract_answer(self, raw_output):
        answer = raw_output.split("Answer: ")[-1]
        last_start = answer.rfind("[START]")  # Find last occurrence of [START]

        if last_start == -1:
            return "Extraction Failed: No [START] marker found."

        last_end = answer.find("[END]", last_start)  # Find first [END] after last [START]

        if last_end == -1:
            return "Extraction Failed: No closing [END] found after last [START]."

        # Extract answer
        answer = answer[last_start + len("[START]"):last_end].strip()

        return answer if answer else "Extraction Failed: Empty answer."