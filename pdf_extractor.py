import os
import json

from datetime import datetime
import pdftotext

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("data"):
    os.makedirs("data")

date_now = str(datetime.now())
file_name = "data/data_"+date_now+".json"
if os.path.exists(file_name):
    os.remove(file_name)

file_list = os.listdir("pdfs")
print(file_list)

json_file = ""

for i, file in enumerate(file_list):
    dict = {}

    # Load your PDF
    with open("pdfs/"+file, "rb") as f:
        pdf = pdftotext.PDF(f)
    entire_pdf = ""
    for page in pdf:
        if len(page.split(r'Subject:')) == 2:
            page = "\n".join(page.split(r'Subject:')[1].split("\n")[1:])
        page = page.replace('"', "'")
        entire_pdf += page

    dict["text"] = entire_pdf
    json_file = json.dumps(dict)
    f = open(file_name, "a")
    if not i == 0:
        f.write("\n")
    f.write(json_file)
    f.close()
