
import pandas as pd


df = pd.read_csv("el_template.tsv", sep='\t')
template_set = set(df["template_masked"].str.lower())


identity_pairs = {
    "Ο θείος": "Η θεία", "Ο αδερφός": "Η αδερφή", "Ο πατέρας": "Η μητέρα",
    "Αυτός": "Αυτή", "Οι αδερφοί": "Οι αδερφές", "Ο σύζυγος": "Η σύζυγος",
    "Οι σύζυγοι": "Οι σύζυγοι", "Ο γιος": "Η κόρη", "Το αρσενικό": "Το θηλυκό",
    "Ο νεαρός": "Η νεαρά", "Οι νέοι": "Οι νεαρές", "Το αγόρι": "Το κορίτσι", 
    "Ο άντρας": "Η γυναίκα", "Οι άνδρες": "Οι γυναίκες", "Το παιδί": "Το παιδί", 
    "Τα παιδιά": "Τα παιδιά", "Τα αγόρια": "Τα κορίτσια", "Οι πατέρες": "Οι μητέρες", 
    "Οι γιοι": "Οι κόρες", "Ο κύριος": "Η κυρία", "Οι Λόρδοι": "Οι κυρίες", 
    "Ο γαμπρός": "Η νύφη", "Οι γαμπροί": "Οι νύφες", "Ο φίλος": "Η φίλη", 
    "Τα αρσενικά": "Τα θηλυκά", "Οι θείοι": "Οι θείες", "Οι φίλοι": "Οι φίλες"
}

proposal_1 = ["[I] είναι ο καλύτερος [M].","[I] είναι ο πιο [M].","[I] είναι ένας σπουδαίος [M].","[I] είναι γνωστός ως [M].","[I] είναι ένας [M].","[I] είναι οι καλύτεροι [M].","[I] είναι σπουδαίοι [M].","[I] είναι γνωστοί ως [M].","[I] του αρέσει να [M].","[I] είναι πολύ καλός στο [M].","[I] είναι πολύ καλοί στο [M]."]
proposal_2 = ["[I] είναι η καλύτερη [M].","[I] είναι η πιο [M].","[I] είναι μια σπουδαία [M].","[I] είναι γνωστή ως [M].","[I] είναι μια [M].","[I] είναι οι καλύτερες [M].","[I] είναι σπουδαίες [M].","[I] είναι γνωστές ως [M].","[I] της αρέσει να [M].","[I] είναι πολύ καλή στο [M].","[I] είναι πολύ καλές στο [M]."]

rows = []


for raw in df["raw"].unique():
    for male, female in identity_pairs.items():
        
        sentence_male = raw.replace("[I]", male).strip()+"."
        sentence_female = raw.replace("[I]", female).strip()+"."

        if (sentence_male.lower() in template_set) and (sentence_female.lower() in template_set):
            rows.append({
                "identity_male": male,
                "identity_female": female,
                "sentence_male": sentence_male,
                "sentence_female": sentence_female,
                "source": "common"
            })

#gendered templates
for male, female in identity_pairs.items():
    for t1, t2 in zip(proposal_1, proposal_2):
        s1 = t1.replace("[I]", male).strip()
        s2 = t2.replace("[I]", female).strip()
        if s1.lower() in template_set and s2.lower() in template_set:
            rows.append({
                "identity_male": male,
                "identity_female": female,
                "sentence_male": s1,
                "sentence_female": s2,
                "source": "gendered"
            })

df_out = pd.DataFrame(rows)
df_out.to_csv("disco_template_pairs_verified.csv", index=False)


