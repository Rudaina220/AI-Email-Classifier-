import os,re,base64,pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
SCOPES=["https://www.googleapis.com/auth/gmail.readonly"]
def auth():
    c=None
    if os.path.exists("token_collect.json"):c=Credentials.from_authorized_user_file("token_collect.json",SCOPES)
    if not c or not c.valid:
        if c and c.expired and c.refresh_token:c.refresh(Request())
        else:c=InstalledAppFlow.from_client_secrets_file("credentials.json",SCOPES).run_local_server(port=0)
        with open("token_collect.json","w") as f:f.write(c.to_json())
    return build("gmail","v1",credentials=c)
def dec(x):
    try:return base64.urlsafe_b64decode(x or "").decode("utf-8",errors="ignore")
    except:return ""
def getbody(p):
    out=[]
    for part in p.get("parts",[]):
        if part.get("mimeType")=="text/plain":out.append(dec(part.get("body",{}).get("data")))
        elif part.get("mimeType","").startswith("multipart/"):out.append(getbody(part))
    if not p.get("parts") and p.get("mimeType")=="text/plain":out.append(dec(p.get("body",{}).get("data")))
    return " ".join(out)
service=auth();res=service.users().messages().list(userId="me",labelIds=["INBOX"],maxResults=200).execute()
rows=[]
for m in res.get("messages",[]):
    msg=service.users().messages().get(userId="me",id=m["id"],format="full").execute();h={}
    for x in msg.get("payload",{}).get("headers",[]):h[x["name"].lower()]=x.get("value","")
    rows.append({"id":m["id"],"subject":h.get("subject",""),"from":h.get("from",""),"body":getbody(msg.get("payload",{}))[:10000],"label":""})
pd.DataFrame(rows).to_csv("gmail_training_raw.csv",index=False)
print("Created gmail_training_raw.csv. Fill label with WORK, PERSONAL, or URGENT, then copy rows into category_training.csv.")
