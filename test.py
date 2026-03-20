import json

with open('data/train_v2_hintscore.json', 'r') as f:
    train_hintscore = json.load(f)
with open('data/test_v2_hintscore.json', 'r') as f:
    test_hintscore = json.load(f)
print(len(train_hintscore),len(test_hintscore))
qids = '265814016'

if qids in train_hintscore.keys():
  print(train_hintscore[qids])
if qids in test_hintscore.keys():
  print(test_hintscore[qids])