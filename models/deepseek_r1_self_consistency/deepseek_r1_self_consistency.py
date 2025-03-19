from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn.functional as F
from collections import Counter


class deepseek_r1_self_consistency:
    def __init__(self, params):
        self.chunk_size = int(params[0])
        self.k_articles = int(params[1])
        self.k_chunks = int(params[2])
        self.m = int(params[3])

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

    def invoke(self, questions):
        contexts = [self.get_context(q) for q in questions]

        messages_batch = [self.create_messages(q, c) for q, c in zip(questions, contexts)]
        return self.get_answer(messages_batch)

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

    def get_answer(self, messages_batch):
        samples = self.sample_answers(messages_batch)
        answers = [self.select_consistent_answer(sample) for sample in samples]
        return answers

    def sample_answers(self, messages_batch):
        samples = []
        for _ in range(self.m):
            responses = self.generate_answer(messages_batch)
            answers = [self.extract_answer(response) for response in responses]
            samples.append(answers)
        samples = [sample for sample in zip(*samples)]
        return samples

    def select_consistent_answer(self, samples):
        # Flatten samples if they are in tuple/list form
        flattened_samples = [" ".join(sample) if isinstance(sample, (tuple, list)) else sample for sample in samples]

        # Count occurrences of each response
        answer_counts = Counter(flattened_samples)

        # Return the most common answer
        most_common_answer, _ = answer_counts.most_common(1)[0]
        return most_common_answer

    def create_messages(self, question, contexts):
        system_prompt = """
        You are an AI assistant that follows a structured chain of thought.
        """

        constraints = """
        STRICT RULES:
        - DO NOT use "<think>" before answering.
        - Use EXACTLY 4 reasoning steps—no more, no less.
        - Each reasoning step must be MAXIMUM 10 words.
        - NO self-reflection, process description, or over-explanation.
        - STOP RESPONDING after providing the final answer.
        """

        examples = """Example:
            **Question:** Who directed the movie La Dolce Vita?
            **Context:** La Dolce Vita is a 1960 Italian film directed by Federico Fellini.
            **Response:**
                1. The question asks for the director of La Dolce Vita.
                2. The context states Federico Fellini directed the movie.
                3. No other director is mentioned.
                4. Federico Fellini is confirmed as the director.
            Final Answer: [START]Federico Fellini[END]

        Second Example:
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Now answer the following question in **the same format as the examples**:"},
            {"role": "user", "content": f"**Question:** {question}"},
            {"role": "user", "content": f"**Context:** {contexts}"},
            {"role": "user", "content": constraints},
            {"role": "user", "content": f"Here are examples of correct responses:\n\n{examples}"},
            {"role": "user", "content": "**Begin Response Below:**\n\n1."}
        ]

        return messages

    def generate_answer(self, messages_batch):
        torch.cuda.empty_cache()  # Free unused GPU memory before inference
        torch.cuda.synchronize()  # Ensure memory is properly released

        input_ids_list = []

        # Encode each message separately and track max sequence length
        max_length = 0

        for messages in messages_batch:
            encoded_input = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                truncation=True,
                max_length=8096,
                padding_side='left'
            )

            # Ensure batch dimension exists
            if encoded_input.dim() == 1:
                encoded_input = encoded_input.unsqueeze(0)

            input_ids_list.append(encoded_input)
            max_length = max(max_length, encoded_input.shape[-1])  # Track longest sequence

        # **Manually pad each tensor to max_length**
        padded_input_ids = torch.stack([
            F.pad(tensor.squeeze(0), (0, max_length - tensor.shape[-1]), value=self.tokenizer.pad_token_id)
            for tensor in input_ids_list
        ]).to(self.device)

        # ✅ Ensure `attention_mask` is **2D** (batch_size, seq_length)
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long().to(self.device)

        # **Debugging Output**
        print(f"input_ids shape: {padded_input_ids.shape}")  # Should be (batch_size, seq_length)
        print(f"attention_mask shape: {attention_mask.shape}")  # Should be (batch_size, seq_length)

        with torch.inference_mode():  # Optimize inference
            outputs = self.model.generate(
                input_ids=padded_input_ids,
                attention_mask=attention_mask,  # ✅ Fixed to 2D
                max_new_tokens=200,  # Reduce token limit for efficiency
                repetition_penalty=1.2,
                temperature=0.3,  # Slightly higher temperature for creativity in reasoning
                top_p=0.9,  # Allow some diversity in reasoning steps
                do_sample=True,  # Enable sampling for varied reasoning
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        torch.cuda.empty_cache()  # Free GPU memory after inference

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