from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch



class deepseek_r1_distill_llama_8B:
    def __init__(self, params):
        login()
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.database = FAISS.load_local("database", self.embedding_model, allow_dangerous_deserialization=True)
        self.device = "cuda:0"

        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.initialize_model(model_name=self.model_name)

        self.k = params[0]

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

    def invoke(self, question):
        context = self.get_context(question)
        return self.get_answer(question, context)

    def get_context(self, question):
        documents = self.database.similarity_search(question, self.k)
        context = ""
        for doc in documents:
            context = context + doc.page_content + "\n"
        return context

    def get_answer(self, question, context):
        messages = self.create_messages(question, context)
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs)
        answer =  self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer[answer.rfind('\n'):]
        return answer

    def create_messages(self, question, contexts):
        question_add = [" To answer the question use the following text to find the answer:",
                        "\nAnswer in one or two words, no additional information, no punctiation. "]
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