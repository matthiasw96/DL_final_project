from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import torch


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
        samples = self.sample_answers(message)
        answer = self.select_answer(samples)
        return answer

    def sample_answers(self, message):
        samples = []
        for _ in range(self.m):
            r = self.generate_answer(message)
            a = self.extract_answer(r)
            samples.append(a)
        return samples

    def select_answer(self, samples):
        answer_embeddings, cluster_groups, cluster_labels = self.cluster_answers(samples)
        answer = self.majority_vote(answer_embeddings, cluster_groups, cluster_labels)
        return samples[answer]

    def cluster_answers(self, samples):
        answer_embeddings = self.embedding_model.embed_documents(samples)
        answer_embeddings = np.array(answer_embeddings)

        distance_matrix = 1 - cosine_similarity(answer_embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.3,
            metric='precomputed',
            linkage='average')

        cluster_labels = clustering.fit_predict(distance_matrix)
        cluster_groups = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            cluster_groups[label].append(samples[idx])
        return answer_embeddings, cluster_groups, cluster_labels

    def majority_vote(self, answer_embeddings, cluster_groups, cluster_labels):
        majority_label = max(cluster_groups, key=lambda k: len(cluster_groups[k]))
        majority_indices = [idx for idx, lbl in enumerate(cluster_labels) if lbl == majority_label]

        majority_embeddings = answer_embeddings[majority_indices]
        centroid = np.mean(majority_embeddings, axis=0)
        similarities = cosine_similarity([centroid], majority_embeddings)[0]
        most_representative_idx = np.argmax(similarities)

        return majority_indices[most_representative_idx]

    def create_messages(self, question, contexts):
        constraints = """
                     ONLY USE A MAXIMUM OF 4 REASONING STEPS! 
                     ONLY WRITE ONE SENTENCE PER REASONING STEP! 
                     MARK YOUR FINAL ANSWER WITH "Final Answer: "! 
                     YOUR FINAL SHOULD BE A SINGLE EXPRESSION, NOT A SENTENCE!"""

        examples = """First Example:
         Question: Who directed the movie La Dolce Vita?
         Context: La Dolce Vita is a 1960 Italian film directed by Federico Fellini. The film won the Palme d'Or and remains one of the most influential films in history.
         Response:
             1. The question asks for the director of La Dolce Vita.
             2. The context states that Federico Fellini directed the movie.
             3. No other director is mentioned.
             4. Since Fellini is the only mentioned director, he is the correct answer.
         Final Answer: Federico Fellini

         Second Example:
         Question: What was Diana Ross's first solo No. 1?
         Context: Diana Ross released her debut solo album in 1970, which contained "Ain’t No Mountain High Enough," her first solo No. 1 hit.
         Response:
             1. The question asks for Diana Ross's first solo No. 1.
             2. The context confirms that "Ain’t No Mountain High Enough" was her first solo No. 1.
             3. No earlier solo No. 1 hits are mentioned.
             4. Since this matches the requirement, it is the correct answer.
         Final Answer: Ain't No Mountain High Enough."""

        # Optimized Prompt Template
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that solves problems step by step using the provided context."},
            {"role": "user",
             "content": "Answer the question based on the context by pointing out each reasoning step."},
            {"role": "user", "content": constraints},
            {"role": "user", "content": f"Here are some examples:\n\n{examples}"},
            {"role": "user", "content": "Now answer the following question:"},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "user", "content": f"Context: {contexts}"}
        ]
        return messages

    def generate_answer(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=800,  # Increase token limit for CoT reasoning
            temperature=0.6,  # High temperature to explore different reasoning paths
            top_p=0.9,  # High diversity to produce various outputs
            do_sample=False,  # Enable sampling for varied reasoning
            eos_token_id=self.tokenizer.eos_token_id
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def extract_answer(self, raw_output):
        answer = raw_output.split("Answer:")[-1]
        if answer.startswith(" "):
            answer = answer.replace(" ", "", 1).strip()
        return answer