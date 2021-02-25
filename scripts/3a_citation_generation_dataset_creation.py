import json
import os
import tqdm
from tqdm import tqdm 

    ###############################################################################
if __name__ == "__main__":
    paper_ids = set()
    with open("./notebooks/paper_ids.txt", "r") as f:
        for v in f: paper_ids.add(v.strip("\n"))
    print(f"Number of paper_ids: {len(paper_ids)}")
    
    with open("/home/jessica/data/s2orc/test_result_subset.jsonl", "w") as writefile:
        