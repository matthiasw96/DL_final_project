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
          Role: You are an AI assistant that always answers a question with one expression instead of a full sentence
          <|start_header_id|>user<|end_header_id|>
          Task: Answer the question based on the context only using one expression instead of a full sentence

          Here are some examples:
          First Example
          Question: Tony Lumpkin, Constance Neville and George Hastings are all characters in which play by Irish author Oliver Goldsmith?
          Context: She Stoops to Conquer is a comedy by Anglo-Irish  author Oliver Goldsmith that was first performed in London in 1773. The play is a favourite for study by English literature and theatre classes in the English-speaking world. It is one of the few plays from the 18th century to have an enduring appeal, and is still regularly performed today. It has been adapted into a film several times, including in 1914 and 1923.\n\n*Miss Constance Neville – Niece of Mrs. Hardcastle, she is the woman whom Hastings intends to court. Constance despises her cousin Tony, she is heir to a large fortune of jewels, hence her aunt wants her to remain in the family and marry Tony; she is secretly an admirer of George Hastings however. Neville schemes with Hastings and Tony to get the jewels so she can then flee to France with her admirer; this is essentially one of the sub-plots of She Stoops to Conquer.\n\n*Tony Lumpkin – Son of Mrs Hardcastle and stepson to Mr Hardcastle, Tony is a mischievous, uneducated playboy. Mrs. Hardcastle has no authority over Tony, and their relationship contrasts with that between Hardcastle and Kate. He is promised in marriage to his cousin, Constance Neville, yet he despises her and thus goes to great effort to help her and Hastings in their plans to leave the country. He cannot reject the impending marriage with Constance, because he believes he's not of age. Tony\n\nThe Unity of Action – This is the one Unity that Goldsmith does not rigorously follow; the inclusion of the subplot of Constance-Hastings eloping distracts from the main narrative of the play. However, it shares similar themes of relationships and what makes the best kind (mutual attraction or the arrangement of a parent or guardian). Furthermore, the subplot interweaves with the main plot, for example when Hastings and Marlow confront Tony regarding his mischief making.\n\n*George Hastings – Friend of Charles Marlow and the admirer of Miss Constance Neville. Hastings is an educated man who cares deeply about Constance, with the intention of fleeing to France with her. However the young woman makes it clear that she can't leave without her jewels, which are guarded by Mrs Hardcastle, thus the pair and Tony collaborate to get hold of the jewels. When Hastings realises the Hardcastle house isn't an inn, he decides not to tell Marlow who would thus leave the premisesimmediately.
          Answer: She Stoops to Conquer
          
          Second Example
          Question: In Greek mythology, what is the name of the giant watchman with 100 eyes, also adopted as the name of a UK retail chain
          Context: Argus Panoptes (or Argos) is a 100-eyed giant in Greek mythology.\n\nMythology\n\nArgus Panoptes (), guardian of the heifer-nymph Io and son of Arestor,  was a primordial giant whose epithet, "Panoptes", "all-seeing", led to his being described with multiple, often one hundred, eyes. The epithet Panoptes was applied to the Titan of the Sun, Helios, and was taken up as an epithet by Zeus, Zeus Panoptes. "In a way," Walter Burkert observes, "the power and order of Argos the city are embodied in Argos the neatherd, lord of the herd and lord of the land, whose name itself is thename of the land."\n\nThe sacrifice of Argus liberated Io and allowed her to wander the earth, although tormented by a gadfly sent by Hera.\n\nAccording to Ovid, to commemorate her faithful watchman, Hera had the hundred eyes of Argus preserved forever, in a peacock\'s tail.  \n\nThe myth makes the closest connection of Argus, the neatherd, with the bull. In the  Library of pseudo-Apollodorus, "Argos killed the bull that ravaged Arcadia, then clothed himself in its skin." \n\nIn popular culture\n\nIn the 5th century and later, Argus\' wakeful alertness was explained for an increasingly literal culture as his having so many eyes that only a few of the eyes would sleep at a time: there were always eyes still awake. In the 2nd century AD Pausanias noted at Argos, in the temple of Zeus Larissaios, an archaic image of Zeus with a third eye in the center of his forehead, allegedly Priam\'s Zeus Herkeios purloined from Troy.\n\n* The fifteenth colossus from the video game Shadow of the Colossus is called Argus and nicknamed "The Sentinel" and "Vigilant Guard". The hundreds of eyes carved into the temple that he resides in refers to the omnividence (all-seeing ability) of Argus Panoptes and the watchful colossus himself.\n* Argus Panoptes served as the inspiration for one of the Kaijin from Kamen Rider Wizard, the Phantom Argos.
          Answer: Argus Panoptes

          Third Example
          Question: On a computer keyboard which letter on the same line is between C and B?
          Context: that and the letter to its right (usually Z or Y). Also the enter key is usually shaped differently. Computer keyboards are similar to electric-typewriter keyboards but contain additional keys, such as the command or Windows keys. There is no standard computer keyboard, although many manufacture imitate the keyboard of PCs. There are actually three different PC keyboard: the original PC keyboard with 84 keys, the AT keyboard also with 84 keys and the enhanced keyboard with 101 keys. The threediffer some what in the placement of function keys, the control keys, the return key, and the shift key.\n\nIn computing, a computer keyboard is a typewriter-style device which uses an arrangement of buttons or keys to act as a mechanical lever or electronic switch. Following the decline of punch cards and paper tape, interaction via teleprinter-style keyboards became the main input device for computers.\n\nIn bilingual regions of Canada and in the French-speaking province of Québec, keyboards can often be switched between an English and a French-language keyboard; while both keyboards share the same QWERTY alphabetic layout, the French-language keyboard enables the user to type accented vowels such as "é" or "à" with a single keystroke. Using keyboards for other languages leads to a conflict: the image on the key does not correspond to the character. In such cases, each new language may require\n\nStandard alphanumeric keyboards have keys that are on three-quarter inch centers (0.750\xa0inches, 19.05\xa0mm), and have a key travel of at least 0.150\xa0inches (3.81\xa0mm). Desktop computer keyboards, such as the 101-key US traditional keyboards or the 104-key Windows keyboards, include alphabetic characters, punctuation symbols, numbers and a variety of function keys. The internationally common 102/104 key keyboards have a smaller left shift key and an additional key with some more symbols between\n\nthat can be controlled with the mouse. In software keyboards, the mouse has to be maneuvered onto the on-screen letters given by the software. On the click of a letter, the software writes the respective letter on the respective spot.Projection (as by laser).
          Answer: V
          
          Now answer the following question using the given context only using one expression instead of a full sentence:
          Question: {question}
          Context: {contexts}
          
          ANSWER IN ONE EXPRESSION. DO NOT WRITE A FULL SENTENCE. DO NOT USE PUNCTUATION.
          <|start_header_id|>assistant<|end_header_id|>"""
        return message

    def generate_answer(self, message):
        inputs = self.tokenizer(message, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return raw_output

    def extract_answer(self, raw_output):
        answer = raw_output.split("Answer: ")[-1]
        if answer.startswith(" "):
            answer = answer.replace(" ", "", 1).strip()
        elif "." in answer:
            answer = answer.replace(".","", 1).strip()
        return answer