import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from collections import defaultdict
from scipy.stats import chisquare
import numpy as np


df = pd.read_csv("disco_template_pairs_verified.csv")

name="nlpaueb/bert-base-greek-uncased-v1"#"xlm-roberta-base" #
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=3)

#lemma
def greek_lemmatize(word):
    word = word.lower().strip()
    endings = ["ος", "η", "ες", "οι", "α", "ού","ών", "ους", "ων", "ου", "ας", "ης","ός","ή","ές","οί","ά","ου","ούς","ού","ής","ρια","ο","ό","ά"]
    for suffix in endings:
        if word.endswith(suffix) and len(word) > 4:
            return word[:-len(suffix)]
    return word

lemma_counts = defaultdict(lambda: {"male": 0, "female": 0})
prediction_log=[]
for _, row in df.iterrows():
    male_sent = row["sentence_male"].replace("[M]", tokenizer.mask_token)
    female_sent = row["sentence_female"].replace("[M]", tokenizer.mask_token)

    preds_male = fill_mask(male_sent)
    preds_female = fill_mask(female_sent)

    words_male = [p["token_str"].strip().lower() for p in preds_male]
    words_female = [p["token_str"].strip().lower() for p in preds_female]

    male_lemmas = [greek_lemmatize(p["token_str"]) for p in preds_male]
    female_lemmas = [greek_lemmatize(p["token_str"]) for p in preds_female]

    for lemma in male_lemmas:
        lemma_counts[lemma]["male"] += 1
    for lemma in female_lemmas:
        lemma_counts[lemma]["female"] += 1

    prediction_log.append({
        "sentence_male": row["sentence_male"],
        "sentence_female": row["sentence_female"],
        "predictions_male": ", ".join(words_male),
        "predictions_female": ", ".join(words_female)
    })

results = []
for lemma, counts in lemma_counts.items():
    m, f = counts["male"], counts["female"]
    if m + f >= 5:
        total = m + f
        expected = [0.5 * total, 0.5 * total]
        observed = [m, f]
        chi, p = chisquare(observed, expected)
        results.append({
            "lemma": lemma,
            "male_count": m,
            "female_count": f,
            "p_value": p,
            "biased": int(p < 0.05)
        })


biased_total = sum(r["biased"] for r in results)
total = len(results)
print(f"DisCo Score (lemmatized): {biased_total / total:.4f}")
pd.DataFrame(prediction_log).to_csv("bert_lemma_disco_predictions.csv", index=False)
pd.DataFrame(results).to_csv("bert_lemma_disco_bias_scores.csv", index=False)
