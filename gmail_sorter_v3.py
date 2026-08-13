import os,re,html,base64,pickle,joblib,numpy as np
from bs4 import BeautifulSoup
from scipy.sparse import hstack,csr_matrix
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES=["https://www.googleapis.com/auth/gmail.modify"]
CREDENTIALS_PATH="credentials.json"
TOKEN_PATH="token.json"
SPAM_MODEL_PATH="spam_model_v2.pkl"
TFIDF_PATH="tfidf_v2.pkl"
SCALER_PATH="scaler_v2.pkl"
CATEGORY_MODEL_PATH="category_model.pkl"
CATEGORY_VECTORIZER_PATH="category_vectorizer.pkl"
MAX_EMAILS=50
SPAM_CONFIDENCE=0.90

TRUSTED_DOMAINS={"zewailcity.edu.eg","kaggle.com","canva.com","deeplearning.ai","roboflow.com","glassdoor.com","wuzzuf.net","openai.com","digitalocean.com"}

def clean_text(text):
    text=str(text)
    text=BeautifulSoup(text,"html.parser").get_text(" ")
    text=html.unescape(text)
    text=re.sub(r"http\S+|www\S+"," URL ",text)
    text=re.sub(r"\S+@\S+"," EMAIL ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip().lower()

def authenticate():
    creds=None
    if os.path.exists(TOKEN_PATH):
        creds=Credentials.from_authorized_user_file(TOKEN_PATH,SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow=InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH,SCOPES)
            creds=flow.run_local_server(port=0)
        with open(TOKEN_PATH,"w") as f:f.write(creds.to_json())
    return build("gmail","v1",credentials=creds)

def decode_data(data):
    if not data:return ""
    try:return base64.urlsafe_b64decode(data).decode("utf-8",errors="ignore")
    except:return ""

def extract_body(payload):
    plain=[];html_parts=[]
    if "parts" in payload:
        for part in payload["parts"]:
            mime=part.get("mimeType","")
            if mime=="text/plain":plain.append(decode_data(part.get("body",{}).get("data")))
            elif mime=="text/html":html_parts.append(decode_data(part.get("body",{}).get("data")))
            elif mime.startswith("multipart/"):
                nested=extract_body(part);plain.append(nested["plain"]);html_parts.append(nested["html"])
    else:
        mime=payload.get("mimeType","");data=decode_data(payload.get("body",{}).get("data"))
        if mime=="text/plain":plain.append(data)
        elif mime=="text/html":html_parts.append(data)
    return {"plain":" ".join(plain),"html":" ".join(html_parts)}

def get_email(service,message_id):
    msg=service.users().messages().get(userId="me",id=message_id,format="full").execute()
    headers={}
    for h in msg.get("payload",{}).get("headers",[]):headers[h.get("name","").lower()]=h.get("value","")
    body=extract_body(msg.get("payload",{}))
    text=body["plain"] or BeautifulSoup(body["html"],"html.parser").get_text(" ")
    return {"id":message_id,"subject":headers.get("subject",""),"from":headers.get("from",""),"body":text,"headers":headers}

def get_inbox(service):
    response=service.users().messages().list(userId="me",labelIds=["INBOX"],maxResults=MAX_EMAILS).execute()
    return [get_email(service,m["id"]) for m in response.get("messages",[])]

def get_sender_email(sender):
    match=re.search(r"<([^>]+)>",sender)
    return (match.group(1) if match else sender).lower().strip()

def get_domain(sender):
    email=get_sender_email(sender)
    return email.split("@",1)[1] if "@" in email else ""

def spam_features(email):
    text=email["subject"]+" "+email["body"]
    urls=len(re.findall(r"http\S+|www\S+",text,re.I))
    emails=len(re.findall(r"\S+@\S+",text))
    phones=len(re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)",text))
    html_flag=int(bool(re.search(r"<html|<body|<div|<table",email["body"],re.I)))
    tracking=int(bool(re.search(r"(utm_|tracking|pixel|click|unsubscribe)",text,re.I)))
    received=sum(1 for k in email["headers"] if k=="received" or k.startswith("received-"))
    attachments=int("attachment" in text.lower())
    return [received,urls,emails,phones,tracking,attachments,html_flag,0,0,0,0,0,0,0]

def predict_spam(model,vectorizer,scaler,email):
    text=clean_text(email["subject"]+" "+email["body"])
    text_vector=vectorizer.transform([text])
    meta=csr_matrix(scaler.transform(np.array(spam_features(email),dtype=float).reshape(1,-1)))
    features=hstack([text_vector,meta])
    pred=int(model.predict(features)[0]);probs=model.predict_proba(features)[0]
    return ("SPAM" if pred==1 else "HAM"),float(max(probs))

def category_fallback(email):
    text=(email["subject"]+" "+email["body"]).lower()
    urgent=["urgent","asap","immediately","action required","deadline","final notice","expires today","security alert","verify your account","password","passkey","suspicious activity"]
    work=["meeting","project","internship","interview","job","application","resume","work","team","manager","course","university","registrar","exam","training"]
    if any(x in text for x in urgent):return "URGENT",0.70
    if any(x in text for x in work):return "WORK",0.65
    return "PERSONAL",0.55

def predict_category(email):
    if not os.path.exists(CATEGORY_MODEL_PATH):return category_fallback(email)
    model=joblib.load(CATEGORY_MODEL_PATH);vectorizer=joblib.load(CATEGORY_VECTORIZER_PATH)
    X=vectorizer.transform([clean_text(email["subject"]+" "+email["body"])])
    return model.predict(X)[0],float(max(model.predict_proba(X)[0]))

def get_or_create_label(service,name):
    labels=service.users().labels().list(userId="me").execute().get("labels",[])
    for label in labels:
        if label["name"].lower()==name.lower():return label["id"]
    return service.users().labels().create(userId="me",body={"name":name,"labelListVisibility":"labelShow","messageListVisibility":"show"}).execute()["id"]

def apply_label(service,message_id,label_id):
    service.users().messages().modify(userId="me",id=message_id,body={"addLabelIds":[label_id]}).execute()

def main():
    print("Loading models...")
    with open(SPAM_MODEL_PATH,"rb") as f:spam_model=pickle.load(f)
    with open(TFIDF_PATH,"rb") as f:vectorizer=pickle.load(f)
    with open(SCALER_PATH,"rb") as f:scaler=pickle.load(f)
    print("Connecting to Gmail...");service=authenticate();print("Gmail connected.")
    label_ids={x:get_or_create_label(service,"AI/"+x) for x in ["SPAM","WORK","PERSONAL","URGENT"]}
    emails=get_inbox(service);print("Emails found:",len(emails))
    for email in emails:
        try:
            domain=get_domain(email["from"])
            if any(domain==trusted or domain.endswith("."+trusted) for trusted in TRUSTED_DOMAINS):spam_label,spam_conf="HAM",1.0
            else:spam_label,spam_conf=predict_spam(spam_model,vectorizer,scaler,email)
            print("\n"+"-"*60);print("Subject:",email["subject"]);print("From:",email["from"]);print("Spam:",spam_label,f"({spam_conf:.2%})")
            if spam_label=="SPAM" and spam_conf>=SPAM_CONFIDENCE:final_label,category_conf="SPAM",spam_conf
            else:final_label,category_conf=predict_category(email)
            print("Final:",final_label,f"({category_conf:.2%})")
            if final_label=="SPAM" or category_conf>=0.60:
                apply_label(service,email["id"],label_ids[final_label]);print("Label applied:","AI/"+final_label)
            else:print("Skipped: low confidence")
        except Exception as e:print("Error:",e)
    print("\nFinished.")

if __name__=="__main__":main()
