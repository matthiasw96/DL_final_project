from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class deepseek_r1_chain_of_thoughts:
    def __init__(self, params):
        self.chunk_size = int(params[0])
        self.k_articles = int(params[1])
        self.k_chunks = int(params[2])

        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.database = FAISS.load_local("database", self.embedding_model, allow_dangerous_deserialization=True)
        self.device = "cuda:0"

        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.initialize_model(model_name=self.model_name)
        self.model.config.use_cache = True
        self.tokenizer.padding_side = "left"

        self.text_splitter = self.create_text_splitter()

    def initialize_model(self, model_name):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            use_flash_attention_2=True,
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

    def invoke(self, questions):
        contexts = [self.get_context(q) for q in questions]

        messages_batch = [self.create_messages(q, c) for q, c in zip(questions, contexts)]
        raw_outputs = self.generate_answer(messages_batch)

        return [self.extract_answer(o) for o in raw_outputs]

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


    def create_messages(self, question, contexts):
        system_prompt = """
        You are an AI assistant that follows a structured chain of thought.
        - DO NOT use "<think>" or reflect on your process.
        - Answer with EXACTLY 4 reasoning steps.
        - Each reasoning step MUST be a bullet point (10 words max).
        - Mark your final answer with "Final Answer: ".
        - The final answer must be ONE concise expression (not a sentence).
        - STOP RESPONDING once the final answer is given.
        """

        constraints = """
        STRICT RULES:
        - DO NOT use "<think>" before answering.
        - Use EXACTLY 4 reasoning steps—no more, no less.
        - Each reasoning step must be MAXIMUM 10 words.
        - NO self-reflection, process description, or over-explanation.
        - STOP RESPONDING after providing the final answer.
        """

        examples = """First Example:
            **Question:** Who directed the movie La Dolce Vita?
            **Context:** La Dolce Vita is a 1960 Italian film directed by Federico Fellini.
            **Response:**
                1. The question asks for the director of La Dolce Vita.
                2. The context states Federico Fellini directed the movie.
                3. No other director is mentioned.
                4. Federico Fellini is confirmed as the director.
            Final Answer: [START]Federico Fellini[END]

        Second Example:
            **Question:** What was Diana Ross's first solo No. 1?
            **Context:** Diana Ross released her debut solo album in 1970, which contained "Ain’t No Mountain High Enough," her first solo No. 1 hit.
            **Response:**
                1. The question asks for Diana Ross's first solo No. 1.
                2. The context confirms "Ain’t No Mountain High Enough" was her first.
                3. No earlier solo No. 1 hits are mentioned.
                4. This matches the requirement for the correct answer.
            Final Answer: [START]Ain't No Mountain High Enough[END]

        Third Example:
            **Question:** What is the capital of Spain?
            **Context:** Madrid is mentioned but not confirmed as the capital.
            **Response:**
                1. The question asks for Spain’s capital.
                2. The context does not confirm it.
                3. Prior knowledge confirms Madrid is the capital.
                4. Madrid is the correct answer.
            Final Answer: [START]Madrid[END]
        """

        # Message-based prompt structure
        messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": "Answer the question based on the given context ONLY. If context is missing, use prior knowledge. DO NOT provide explanations beyond the reasoning steps."},
            {"role": "user", "content": constraints},
            {"role": "user", "content": f"Here are examples of correct responses:\n\n{examples}"},
            {"role": "user", "content": "Now answer the following question in the same format:"},
            {"role": "user", "content": f"**Question:** {question}"},
            {"role": "user", "content": f"**Context:** {contexts}"},
            {"role": "user", "content": "**Response:**\n1."}  # Forces bullet-point start
        ]

        return messages

    def generate_answer(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_generation_prompt=True
        ).to(self.device)

        attention_mask = inputs.attention_mask if hasattr(inputs, "attention_mask") else None

        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=1000,  # Increase token limit for CoT reasoning
            temperature=0.2,  # Slightly higher temperature for creativity in reasoning
            top_p=0.8,  # Allow some diversity in reasoning steps
            do_sample=True,  # Enable sampling for varied reasoning
            use_cache=True,
        )

        torch.cuda.empty_cache()

        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

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