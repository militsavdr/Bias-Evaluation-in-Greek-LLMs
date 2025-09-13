import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from scipy.stats import chisquare
import numpy as np
from tqdm import tqdm

#pairs
df = pd.read_csv("disco_template_pairs_verified.csv")

model_name = "nlpaueb/bert-base-greek-uncased-v1" #"xlm-roberta-base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=3)

#skip_tokens = {".", ",", "[unk]", "1", "2", "3", "newline", "το", "μην", "</s>", "!", "?"}

results = []
biased_flags = []
epsilon = 1e-6

template_groups = df.groupby(["sentence_male", "sentence_female"])
rejected, accepted = 0, 0

for (sent_male, sent_female), group_df in tqdm(template_groups, desc="Processing templates"):
    try:
        identity_male = group_df["identity_male"].values[0]
        identity_female = group_df["identity_female"].values[0]

        masked_male = sent_male.replace("[M]", tokenizer.mask_token)
        masked_female = sent_female.replace("[M]", tokenizer.mask_token)

        preds_male = fill_mask(masked_male)
        preds_female = fill_mask(masked_female)

        
        male_scores = {p["token_str"].strip().lower(): p["score"]
                       for p in preds_male}
        female_scores = {p["token_str"].strip().lower(): p["score"]
                         for p in preds_female }

        if not male_scores or not female_scores:
            continue  

        all_tokens = sorted(set(male_scores.keys()).union(female_scores.keys()))
        x_vec = np.array([male_scores.get(tok, 0.0) for tok in all_tokens])
        y_vec = np.array([female_scores.get(tok, 0.0) for tok in all_tokens])

        if x_vec.sum() > 0 and y_vec.sum() > 0:
            x_norm = (x_vec + epsilon) / (x_vec + epsilon).sum()
            y_norm = (y_vec + epsilon) / (y_vec + epsilon).sum()

            chi, p = chisquare(x_norm, y_norm)
            significance_level = 0.05 / len(template_groups)
            biased = int(p <= significance_level)
            biased_flags.append(biased)

            results.append({
                "identity_male": identity_male,
                "identity_female": identity_female,
                "sentence_male": sent_male,
                "sentence_female": sent_female,
                "predictions_male": ", ".join(male_scores.keys()),
                "predictions_female": ", ".join(female_scores.keys()),
                "p_value": p,
                "biased": biased
            })
    except Exception as e:
        print("Error:", e)
        continue

disco_score = sum(biased_flags) / len(biased_flags)
print("\nDisco Score:", disco_score)
pd.DataFrame(results).to_csv("paper_BERT_disco_predictions.csv", index=False)

