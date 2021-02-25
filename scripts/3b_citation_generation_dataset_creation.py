import json
import os
import tqdm
from tqdm import tqdm 

import pandas as pd

    ###############################################################################
if __name__ == "__main__":
    paper_ids = set()
    with open("./notebooks/paper_ids.txt", "r") as f:
        for v in f: paper_ids.add(v.strip("\n"))
    print(f"Number of paper_ids: {len(paper_ids)}")
    
    result_comps = []
    with open("../misc/result_comps.jsonl", "r") as f:
        for line in tqdm(f):
            p = json.loads(line)
            result_comps.append(p)

            if len(result_comps) > 1000: 
                break
    result_comps = pd.DataFrame(test_result_comps)
    
    with open("/home/jessica/data/s2orc/s2orc_result_subset.jsonl", "r") as f:
        for line in tqdm(f):
            p = json.loads(line)
            s2orc_result_subset[p["paper_id"]] = p["body_text"]

    
    test_result_comps["cited_fullbody"] = test_result_comps.cited_id.apply(lambda i: test_result_subset[i])