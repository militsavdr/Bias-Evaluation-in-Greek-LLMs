import pandas as pd
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

file1_path = "el1_honest_dataset_with_id.tsv"
file2_path = "el3_honest_dataset_with_id.tsv"
output_file = "honest_similarity_results_EN_ES.tsv"

df1 = pd.read_csv(file1_path, sep="\t")
df2 = pd.read_csv(file2_path, sep="\t")

#check order
df1 = df1.sort_values("1").reset_index(drop=True)
df2 = df2.sort_values("1").reset_index(drop=True)

# semantic similarity
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

comparison_results = []

#compare each row
for i in range(len(df1)):
    #loc[row,column]
    id_value = df1.loc[i, "1"]
    sentence1 = df1.loc[i, "template_masked"]
    sentence2 = df2.loc[i, "template_masked"]
    raw1 = df1.loc[i, "raw"]
    raw2 = df2.loc[i, "raw"]
    identity1 = df1.loc[i, "identity"]
    identity2 = df2.loc[i, "identity"]

    # Exact match (with fuzzy matching)
    exact_match_template = 1 if fuzz.ratio(sentence1, sentence2) == 100 else 0
    exact_match_raw = 1 if fuzz.ratio(raw1, raw2) == 100 else 0

    # Semantic similarity (BERT-based)
    embedding1_template = model.encode(sentence1, convert_to_tensor=True)
    embedding2_template = model.encode(sentence2, convert_to_tensor=True)
    similarity_template = util.pytorch_cos_sim(embedding1_template, embedding2_template).item()

    embedding1_raw = model.encode(raw1, convert_to_tensor=True)
    embedding2_raw = model.encode(raw2, convert_to_tensor=True)
    similarity_raw = util.pytorch_cos_sim(embedding1_raw, embedding2_raw).item()

    # if similarity > 0.9 then match 
    semantic_match_template = 1 if similarity_template > 0.9 else 0
    semantic_match_raw = 1 if similarity_raw > 0.9 else 0

    final_match = 0 if ((exact_match_template == 1 or semantic_match_template == 1) and
                        (exact_match_raw == 1 or semantic_match_raw == 1)) else 1

    #add to list
    comparison_results.append({
        "1": id_value,
        "template_masked_1": sentence1,
        "template_masked_2": sentence2,
        "raw_1": raw1,
        "raw_2": raw2,
        "identity_1": identity1,
        "identity_2": identity2,
        "similarity_template": round(similarity_template, 3),
        "similarity_raw": round(similarity_raw, 3),
        "match": final_match
    })

# convert to DataFrame
df_results = pd.DataFrame(comparison_results)

df_results.to_csv(output_file, sep="\t", index=False)

print(f"saved as {output_file}")
