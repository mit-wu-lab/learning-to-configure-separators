import numpy as np
import argparse
import os

def get_action_space(args):
    path_prefix = f"{args.path_to_space_dir}/{args.instances}/"
    spaces = ["near_zero", "near_mask",  "mask_space"]
    all_actions = []
    all_scores = []
    all_scores_detail = []
    for space in spaces:
        path_to_mask = path_prefix + space + "/configs.npy"
        path_to_score = path_prefix + space + "/stats.npy"
        
        if os.path.exists(path_to_mask) == False:
            continue
        
        masks = np.load(path_to_mask)
        scores = np.load(path_to_score)
                
        all_actions += masks.tolist()
        all_scores += scores.mean(axis=1).tolist()
        all_scores_detail += scores.tolist()
    rank = np.argsort(-np.array(all_scores))
    actions = []
    return_scores = []
    for i in range(min(args.action_space_size, len(all_actions))):
        actions.append(np.array(all_actions[rank[i]]).reshape(-1,1))
        return_scores.append(np.array(all_scores_detail[rank[i]]))
    return actions, return_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str, help="the MILP class")
    parser.add_argument("--action_space_size", type=int, default=999999999, help="the size of the action space")
    parser.add_argument('--path_to_space_dir', type=str, default="../restricted_space/", help="the directory of the subspace")
    args = parser.parse_args()
    actions, return_scores = get_action_space(args)
    np.save(f"{args.path_to_space_dir}/{args.instances}/action_space.npy", np.array(actions))
    np.save(f"{args.path_to_space_dir}/{args.instances}/action_scores.npy", np.array(return_scores))