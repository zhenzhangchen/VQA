import json  

# with open('data/vqacp_v2_train_annotations.json','r') as f:
#     train_qid2type = json.load(f) # questionId : questionType

# output_data={}

# with open('util/qid2type_cpv2.json', 'r') as f:
#     exist = json.load(f)
# for item in train_qid2type:
#     key = item['question_id']
#     val = item['answer_type']
#     exist[key]=val

# # # 写入新的 JSON 文件  
# with open('output.json', 'w') as outfile:  
#     json.dump(exist, outfile, ensure_ascii=False, indent=4) 

# print("转换完成，新的 JSON 文件已写入 output.json")  



with open('output.json','r') as f:
    train_qid2type = json.load(f) # questionId : questionType
print(len(train_qid2type))