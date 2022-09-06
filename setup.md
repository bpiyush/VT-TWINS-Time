* Setup `conda` environment
* Download checkpoints, tokenizers, etc.
```sh
# word2vec
ID=1fEPmgV5nJZl6hJYEWvy92rJPbxLYGKam
gdown --id $ID -O ./data/

# dict
ID=1TBRmxEmANrOPO0HlBkKoYS65Q0Lk4ixZ
gdown --id $ID -O ./data/

# checkpoint
ID=10khxtVotQV_izhzVrp7oBeerwTAWIpYV
mkdir -p ./checkpoint/
gdown --id $ID -O ./checkpoint/
```