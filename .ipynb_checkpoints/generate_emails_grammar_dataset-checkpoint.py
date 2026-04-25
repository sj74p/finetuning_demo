import json

data=[
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I am not able attend meeting today because have another work.",
    "output": "I am unable to attend the meeting today because I have another work commitment."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please check this and revert back.",
    "output": "Please review this and get back to me."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I have send the file yesterday.",
    "output": "I sent the file yesterday."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Can you update me once done it.",
    "output": "Can you update me once it is done?"
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We didn't got the response yet.",
    "output": "We have not received a response yet."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I am working on this from yesterday.",
    "output": "I have been working on this since yesterday."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please find the attached file and let me know any changes.",
    "output": "Please find the attached file and let me know if any changes are needed."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I didn't understood the requirement.",
    "output": "I did not understand the requirement."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "He didn't shared the details.",
    "output": "He did not share the details."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We are discussing about this issue.",
    "output": "We are discussing this issue."
  },

  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I will update you once I will complete.",
    "output": "I will update you once I complete it."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please send me the details what we discussed.",
    "output": "Please send me the details we discussed."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "There is many issues in the file.",
    "output": "There are many issues in the file."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I didn't got any update.",
    "output": "I did not receive any updates."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Can you please explain me this?",
    "output": "Can you please explain this to me?"
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We have completed the work but not tested yet.",
    "output": "We have completed the work but have not tested it yet."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please update the document and share back.",
    "output": "Please update the document and share it back."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I am not sure about this can you check.",
    "output": "I am not sure about this. Could you please check?"
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "The file is not uploaded properly please check.",
    "output": "The file was not uploaded properly. Please check."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We need to discuss on this.",
    "output": "We need to discuss this."
  },

  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I will send you once it will ready.",
    "output": "I will send it to you once it is ready."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please confirm me once done.",
    "output": "Please confirm once it is done."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We didn't completed this task.",
    "output": "We did not complete this task."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I am having a doubt in this.",
    "output": "I have a question about this."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "He didn't came to meeting.",
    "output": "He did not come to the meeting."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We are waiting for your reply from yesterday.",
    "output": "We have been waiting for your reply since yesterday."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please do the needful and update me.",
    "output": "Please take the necessary action and update me."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I didn't saw this earlier.",
    "output": "I did not see this earlier."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "The data is not matching please verify.",
    "output": "The data does not match. Please verify."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We need to fix this issue immediately otherwise it will create problem.",
    "output": "We need to fix this issue immediately; otherwise, it will create problems."
  },

  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I will get back to you after checking it once.",
    "output": "I will get back to you after reviewing it."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please let me know if any queries.",
    "output": "Please let me know if you have any questions."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We have not received any update till now.",
    "output": "We have not received any updates so far."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I am sending you the file which you asked.",
    "output": "I am sending you the file you requested."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please check and let me know your feedbacks.",
    "output": "Please review and let me know your feedback."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We didn't got chance to discuss.",
    "output": "We did not get a chance to discuss this."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "I am not able to login from morning.",
    "output": "I have not been able to log in since morning."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "He already completed the task but forgot to update.",
    "output": "He already completed the task but forgot to provide an update."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "Please update once you will done.",
    "output": "Please update once you are done."
  },
  {
    "instruction": "Fix grammar and improve clarity while keeping the meaning the same.",
    "input": "We are facing issue in accessing the file.",
    "output": "We are facing an issue accessing the file."
  }

]

out_filename = "emails_grammar_based.jsonl"

with open(out_filename, "w") as f:
    for example in data:
        record = {
            "instruction": example["instruction"],
            "input": example["input"],
            "output": example["output"]
        }
        f.write(json.dumps(record) + "\n")

print(f"Generated {len(data)} examples in {out_filename}")