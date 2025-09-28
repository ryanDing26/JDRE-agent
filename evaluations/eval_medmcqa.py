#######################################
# Simple evaluation setup for MedMCQA #
#######################################

import re
import json
from tqdm import tqdm
from datasets import load_dataset

def eval_medmcqa(model, retriever, top_k=5):
    dataset = load_dataset("openlifescienceai/medmcqa")
    results = []
    correct = 0

    for row in tqdm(dataset):
        qid = row["id"] # don't really need in all honesty
        q = row["question"]
        a, b, c, d = row["opa"], row["opb"], row["opc"], row["opd"]
        gold = row["cop"]
        subject = row["subject_name"] # useful for subject-level filtering later; lots of null for topic-level though

        ctx_ids = retriever(q, top_k=top_k)
        snippets = [retriever.get_chunk(cid).text for cid in ctx_ids]

        prompt = f"""
            Question: {q}
            
            Subject: {subject}

            Options:

            A. {a}
            B. {b}
            C. {c}
            D. {d}

            Strictly answer by responding with the number corresponding to the correct letter (A: 1, B: 2, C: 3, D: 4). 
            
            For example, if you believe that A is the correct answer, answer with 1 (and so forth for other letters)
        """

        # Generate and parse prediction
        pred = model.generate(prompt)
        # TODO some type checking for this later
        correct += 1 if pred == gold else 0

        results.append(
            {
                "id": qid,
                "question": q,
                "options": [a, b, c, d],
                "gold": gold,
                "pred": pred,
                "raw_pred": pred,
                "correct": 1 if pred == gold else 0,
                "retrieved": snippets
            }
        )

    total_acc = correct / len(dataset)
    return results, total_acc