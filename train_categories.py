import os,re,html,joblib,pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

INPUT_FILE="category_training.csv"
def clean_text(text):
    text=BeautifulSoup(str(text),"html.parser").get_text(" ")
    text=html.unescape(text)
    text=re.sub(r"http\S+|www\S+"," URL ",text)
    text=re.sub(r"\S+@\S+"," EMAIL ",text)
    return re.sub(r"\s+"," ",text).strip().lower()

if not os.path.exists(INPUT_FILE):
    pd.DataFrame({"subject":["Final Exam Schedule","Project meeting","Happy birthday","Security alert"],"body":["University exam schedule","Project meeting tomorrow","Have a great birthday","Verify your account immediately"],"label":["WORK","WORK","PERSONAL","URGENT"]}).to_csv(INPUT_FILE,index=False)
    print("Created category_training.csv. Add at least 30-50 examples per class and run again.")
    raise SystemExit

df=pd.read_csv(INPUT_FILE).dropna(subset=["subject","body","label"])
df["text"]=(df["subject"].astype(str)+" "+df["body"].astype(str)).apply(clean_text)
df["label"]=df["label"].astype(str).str.upper()
df=df[df["label"].isin(["WORK","PERSONAL","URGENT"])]
if len(df)<30 or df["label"].nunique()<3:
    print("Need at least 30 labeled examples total and all three classes.")
    raise SystemExit

Xtr,Xte,ytr,yte=train_test_split(df["text"],df["label"],test_size=.2,random_state=42,stratify=df["label"])
vec=TfidfVectorizer(max_features=100000,ngram_range=(1,2),min_df=1,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr);Xte=vec.transform(Xte)
model=LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(Xtr,ytr);pred=model.predict(Xte)
print("Accuracy:",accuracy_score(yte,pred));print(classification_report(yte,pred))
joblib.dump(model,"category_model.pkl");joblib.dump(vec,"category_vectorizer.pkl")
print("Saved category_model.pkl and category_vectorizer.pkl")
