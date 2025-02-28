from datasets import load_dataset
from tqdm import tqdm
import json
import argparse


def download_dataset(split, split_size):
    evaluation_data = load_dataset('trivia_qa', name='rc.wikipedia.nocontext', split=f"{split}[:{split_size}]")
    return evaluation_data


def format_to_json(data):
    file = "{\"Data\":["
    i = 0
    for i in tqdm(range(len(data)), desc="Loading dataset"):
        file += json.dumps(data[i])
        if i < len(data) - 1:
            file += ",\n"
        else:
            file += "],\n"
        i += 1
    file += "\"Domain\": \"Wikipedia\",\"VerifiedEval\": false,\"Version\": 1.0}"
    return file


def replace_field_names(file):
    old_names = ["normalized_matched_wiki_entity_name", "matched_wiki_entity_name",
                 "normalized_aliases", "normalized_value", "question_source",
                 "search_results", "entity_pages", "description", "question_id",
                 "doc_source", "filename", "question", "aliases", "answer", "value",
                 "title", "rank", "type", "url"]
    new_names = ["NormalizedMatchedWikiEntityName", "MatchedWikiEntityName",
                 "NormalizedAliases", "NormalizedValue", "QuestionSource",
                 "SearchResults", "EntityPages", "Description", "QuestionId", "DocSource",
                 "Filename", "Question", "Aliases", "Answer", "Value", "Title", "Rank",
                 "Type", "Url"]

    for i in tqdm(range(len(old_names)), desc="Replacing field names"):
        file = file.replace(old_names[i], new_names[i])

    return file


def write_file(file):
    print("Saving file...")
    with open(f"evaluation/datasets/validation_dataset.json", "w", encoding="utf-8") as f:
        f.write(file)


def get_args():
    parser = argparse.ArgumentParser(
        description='Dataset Loader for Evaluation Dataset', )
    parser.add_argument('--split', help='split of the triviaqa set')
    parser.add_argument('--split_size', type=int, help='size of the split')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    split = args.split
    split_size = args.split_size
    data = download_dataset(split, split_size)
    data = format_to_json(data)
    data = replace_field_names(data)
    write_file(data)
