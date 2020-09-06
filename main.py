import os
import json
import re

from datetime import datetime

import email
from email import policy
from email.parser import BytesParser

file_paths = os.listdir("odin_mails")

json_file = ""
date_now = str(datetime.now())
file_name = "data_" + date_now + ".json"

for i, path in enumerate(file_paths):
    dict = {}

    file = open("odin_mails/" + path, "rb")
    msg = BytesParser(policy=policy.default).parse(file)
    msg_body = msg.get_body(preferencelist=("plain")).get_content()
    msg_body = re.sub("\n{2,}", "\n\n", msg_body)
    # print("From: " + msg_body.split("From: ")[1])

    dict["text"] = msg_body.split("Onderwerp: ")[1]
    json_file = json.dumps(dict)
    f = open(file_name, "a")
    if not i == 0:
        f.write("\n")
    f.write(json_file)
    f.close()
