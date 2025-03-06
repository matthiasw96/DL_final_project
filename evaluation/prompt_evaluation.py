import json
import math
from random import random
from tqdm import tqdm
import re


class prompt_evaluation:
    def __init__(self, test_model, load_tuples=True):
        self.test_model = test_model
        load_tuples = load_tuples
        if load_tuples:
            self.tuples = self.load_tuples()
        else :
            self.tuples = self.create_tuples()

        self.dataset = self.load_dataset()

    def create_tuples(self):
        tuples = []
        for question in tqdm(self.dataset["Data"]):
            context = self.test_model.get_context(question)
            tuples.append((question,context))
        with open("evaluation/dataset/question_context_tuples.json", "w", encoding="utf-8") as f:
            json.dump(tuples, f)
        return tuples

    def test_prompt(self, prompt_template, confidence, error_rate):
        test_passed = True

        n = math.log(1-confidence)/math.log(1-error_rate)
        samples = self.random_samples(prompt_template, n)

        for sample in samples:
            if not self.check_sample(sample):
                print(sample)
                test_passed = False
        return test_passed

    def random_samples(self, prompt_template, n):
        sample_tuples = []
        samples = []

        while len(sample_tuples) < n:
            sample_tuple = self.tuples[random.randint(0, len(self.tuples)-1)]
            if sample_tuple not in sample_tuples:
                sample_tuples.append(sample_tuple)

        for sample_tuple in tqdm(sample_tuples, desc="Sampling answers..."):
            prompt = prompt_template.format(question=sample_tuple[0], context=sample_tuple[1])
            answer = self.test_model.generate_answer(prompt)
            sample = self.test_model.extract_answer(answer)
            samples.append(sample)
        return samples

    def check_sample(self, sample):
        if re.search(r'[.!?,]', sample):  # If it contains punctuation like . ! ? ,
            return False
        if re.search(r'\b(is|was|were|are|has|have|had|the|a|an|of|in|on|to)\b', sample, re.IGNORECASE):
            return False  # Avoid full sentences
        return True

    def load_dataset(self):
        with open("evaluation/datasets/test_dataset.json", "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        return data

    def load_tuples(self):
        with open("evaluation/datasets/question_context_tuples.json", "r", encoding="utf-8") as f:
            tuples = json.loads(f.read())
        return tuples