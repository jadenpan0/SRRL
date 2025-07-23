import os
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
import torch
from functools import partial
import tqdm
import pickle
import tqdm
import json
from bert_score import score, BERTScorer
from PIL import Image
import open_clip
import numpy as np
from utils.utils import seed_everything

##CLIP Score
def score_fn1(ground, img_dir, save_dir, config): 
    unique_id = config.exp_name

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model = model.to(device)

    eval_list = sorted(os.listdir(img_dir))

    similarity = []
    maximum_onetime = 8
    for i in range(0, len(eval_list), maximum_onetime): 
        image_input = torch.tensor(np.stack([preprocess(Image.open(os.path.join(img_dir, image))).numpy() for image in eval_list[i:i+maximum_onetime]])).to(device)
        text_inputs = tokenizer(ground[i:i+maximum_onetime]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity.append( (image_features @ text_features.T)[torch.arange(maximum_onetime), torch.arange(maximum_onetime)] )
    similarity = torch.cat(similarity)

    R = similarity.cpu().detach()
    print(R[:10])

    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'scores.pkl'),'wb') as f:
        pickle.dump(R, f)

    each_score = {}
    for idx, prompt in enumerate(ground): 
        if prompt in each_score:
            each_score[prompt].append(R[idx:idx+1])
        else: 
            each_score[prompt] = [R[idx:idx+1]]

    history_data = []
    if config.eval.history_cnt > 0:
        if os.path.exists(os.path.join(config.save_path, unique_id, 'history_scores.pkl')):
            with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'rb') as f:
                history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]
    data_mean = {}
    data_std = {}
    cur_data = {}
    combine_data = {}
    for k,v in each_score.items():
        cur_data[k] = torch.cat(v, axis=0)
        combine_data[k] = torch.cat([d[k] for d in history_data if k in d]+[cur_data[k]], axis=0)
        data_mean[k] = combine_data[k].mean().item()
        data_std[k] = combine_data[k].std().item()
    history_data.append(cur_data)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    print("==== history_scores ====")
    for k,v in combine_data.items():
        print(k, v.shape)
    print('history data length', len(history_data))
    with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'wb') as f:
        pickle.dump(history_data, f)

    print(data_mean)

    sum_scores = [(s-data_mean[ground[idx]])/(data_std[ground[idx]]+1e-8) for idx, s in enumerate(R)]

    # print(sum_scores)
    sum_scores = torch.tensor(sum_scores) # , dtype=torch.float16)

    return sum_scores

## ImageReward score
def score_fn2(ground, img_dir, save_dir, config): 
    unique_id = config.exp_name
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    image_reward_score = t2v_metrics.ITMScore(model='image-reward-v1',device=device) 

    # device = f"cuda" if torch.cuda.is_available() else "cpu"
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
    # tokenizer = open_clip.get_tokenizer('ViT-H-14')
    # model = model.to(device)

    eval_list = sorted(os.listdir(img_dir))
    eval_list = [os.path.join(img_dir, image) for image in eval_list]

    similarity = []
    maximum_onetime = 8
    for i in tqdm(range(0, len(eval_list), maximum_onetime)): 

        image_input=  eval_list[i:i+maximum_onetime]
        text_inputs = ground[i:i+maximum_onetime]
        similarity.append(image_reward_score(images=image_input, texts=text_inputs)[torch.arange(maximum_onetime), torch.arange(maximum_onetime)])

    similarity = torch.cat(similarity)

    R = similarity.cpu().detach()
    print(R[:10])

    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'scores.pkl'),'wb') as f:
        pickle.dump(R, f)

    each_score = {}
    for idx, prompt in enumerate(ground): 
        if prompt in each_score:
            each_score[prompt].append(R[idx:idx+1])
        else: 
            each_score[prompt] = [R[idx:idx+1]]

    history_data = []
    if config.eval.history_cnt > 0:
        if os.path.exists(os.path.join(config.save_path, unique_id, 'history_scores.pkl')):
            with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'rb') as f:
                history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]
    data_mean = {}
    data_std = {}
    cur_data = {}
    combine_data = {}
    for k,v in each_score.items():
        cur_data[k] = torch.cat(v, axis=0)
        combine_data[k] = torch.cat([d[k] for d in history_data if k in d]+[cur_data[k]], axis=0)
        data_mean[k] = combine_data[k].mean().item()
        data_std[k] = combine_data[k].std().item()
    history_data.append(cur_data)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    print("==== history_scores ====")
    for k,v in combine_data.items():
        print(k, v.shape)
    print('history data length', len(history_data))
    with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'wb') as f:
        pickle.dump(history_data, f)

    print(data_mean)

    sum_scores = [(s-data_mean[ground[idx]])/(data_std[ground[idx]]+1e-8) for idx, s in enumerate(R)]

    print(sum_scores)
    sum_scores = torch.tensor(sum_scores) # , dtype=torch.float16)

    return sum_scores

## clip_flant5_score
def score_fn3(ground, img_dir, save_dir, config): 
    unique_id = config.exp_name
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl', device=device) 

    # device = f"cuda" if torch.cuda.is_available() else "cpu"
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
    # tokenizer = open_clip.get_tokenizer('ViT-H-14')
    # model = model.to(device)

    eval_list = sorted(os.listdir(img_dir))
    eval_list = [os.path.join(img_dir, image) for image in eval_list]

    similarity = []
    maximum_onetime = 8
    for i in tqdm(range(0, len(eval_list), maximum_onetime)): 

        image_input=  eval_list[i:i+maximum_onetime]
        text_inputs = ground[i:i+maximum_onetime]
        similarity.append(clip_flant5_score(images=image_input, texts=text_inputs)[torch.arange(maximum_onetime), torch.arange(maximum_onetime)])

    similarity = torch.cat(similarity)

    R = similarity.cpu().detach()
    print(R[:10])

    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'scores.pkl'),'wb') as f:
        pickle.dump(R, f)

    each_score = {}
    for idx, prompt in enumerate(ground): 
        if prompt in each_score:
            each_score[prompt].append(R[idx:idx+1])
        else: 
            each_score[prompt] = [R[idx:idx+1]]

    history_data = []
    if config.eval.history_cnt > 0:
        if os.path.exists(os.path.join(config.save_path, unique_id, 'history_scores.pkl')):
            with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'rb') as f:
                history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]
    data_mean = {}
    data_std = {}
    cur_data = {}
    combine_data = {}
    for k,v in each_score.items():
        cur_data[k] = torch.cat(v, axis=0)
        combine_data[k] = torch.cat([d[k] for d in history_data if k in d]+[cur_data[k]], axis=0)
        data_mean[k] = combine_data[k].mean().item()
        data_std[k] = combine_data[k].std().item()
    history_data.append(cur_data)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    print("==== history_scores ====")
    for k,v in combine_data.items():
        print(k, v.shape)
    print('history data length', len(history_data))
    with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'wb') as f:
        pickle.dump(history_data, f)

    print(data_mean)

    sum_scores = [(s-data_mean[ground[idx]])/(data_std[ground[idx]]+1e-8) for idx, s in enumerate(R)]

    print(sum_scores)
    sum_scores = torch.tensor(sum_scores) # , dtype=torch.float16)

    return sum_scores