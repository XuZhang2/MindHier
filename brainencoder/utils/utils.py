import os
import re
import random
import math
import json
import requests
import braceexpand
from PIL import Image
import numpy as np

import torch.utils
import torch.utils.data
import webdataset as wds
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes

try:
    from utils.constants import NSD_NUM_IMAGES
except ImportError:
    from brainencoder.utils.constants import NSD_NUM_IMAGES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_id_bbox_caption(input_str):
    bbox = re.findall(r'(\w+)\[(.*?)\]', input_str)
    caption = re.sub(r'\[\d.*?\]', '', input_str).strip(" <s></s>")
    return bbox, caption

def get_loss_func(recon_loss):
    loss_functions = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'huber': F.smooth_l1_loss,
        'quantile': lambda x, y: torch.quantile(torch.abs(x - y), 0.9)
    }
    if recon_loss not in loss_functions:
        raise ValueError(f"Unrecognized loss type: {recon_loss}")
    return loss_functions[recon_loss]

def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print('Note: not using cudnn.deterministic')

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

def get_huggingface_urls(commit='main', subj=1):
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/"
    train_url = base_url + commit + f"/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar"
    val_url = base_url + commit + f"/webdataset_avg_split/val/val_subj0{subj}_0.tar"
    test_url = base_url + commit + f"/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
    return train_url, val_url, test_url
    
def get_dataloaders(
    batch_size,
    image_var='images',
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    meta_url=None,
    num_train=None,
    num_val=None,
    cache_dir="/tmp/wds-cache",
    voxels_key="nsdgeneral.npy",
    val_batch_size=None,
    to_tuple=["voxels", "images", "trial"],
    subj=1,
    data_ratio=1.0,
):
    print("Getting dataloaders...")
    assert image_var == 'images'
    
    def my_split_by_node(urls):
        return urls
    
    train_url = list(braceexpand.braceexpand(train_url))
    val_url = list(braceexpand.braceexpand(val_url))
    if not os.path.exists(train_url[0]):
        # we will default to downloading from huggingface urls if data_path does not exist
        print("downloading NSD from huggingface...")
        os.makedirs(cache_dir, exist_ok=True)
        
        train_url, val_url, test_url = get_huggingface_urls("main", subj)
        train_url = list(braceexpand.braceexpand(train_url))
        val_url = list(braceexpand.braceexpand(val_url))
        test_url = list(braceexpand.braceexpand(test_url))
        
        from tqdm import tqdm
        for url in tqdm(train_url):
            destination = cache_dir + "/webdataset_avg_split/train/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)
                
        for url in tqdm(val_url):
            destination = cache_dir + "/webdataset_avg_split/val/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)
                
        for url in tqdm(test_url):
            destination = cache_dir + "/webdataset_avg_split/test/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)

    if num_devices is None:
        num_devices = torch.cuda.device_count()
    
    if num_workers is None:
        num_workers = num_devices
    
    if num_train is None:
        metadata = json.load(open(meta_url))
        num_train = metadata['totals']['train']
    if num_val is None:
        metadata = json.load(open(meta_url))
        num_val = metadata['totals']['val']

    if val_batch_size is None:
        val_batch_size = batch_size
        
    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    print("\nnum_train", num_train)
    print("global_batch_size", global_batch_size)
    print("batch_size", batch_size)
    print("num_workers", num_workers)
    print("num_batches", num_batches)
    print("num_worker_batches", num_worker_batches)
    
    num_samples = int(num_train * data_ratio)
    train_data = wds.WebDataset(train_url, resampled=True, cache_dir=cache_dir, nodesplitter=wds.split_by_worker)\
        .shuffle(500, initial=500, rng=random.Random(42))\
        .slice(num_samples)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(batch_size, partial=True)\
        .with_epoch(num_worker_batches)
    
    train_dl = DataLoader(train_data, batch_size=None, num_workers=1, shuffle=False)

    # validation (no shuffling, should be deterministic)  
    num_batches = math.floor(num_val / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    print("\nnum_val", num_val)
    print("val_num_batches", num_batches)
    print("val_batch_size", val_batch_size)
    
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(val_batch_size, partial=False)
    
    val_dl = DataLoader(val_data, batch_size=None, num_workers=1, shuffle=False)

    return train_dl, val_dl, num_train, num_val

from torch.utils.data import Dataset

class NSDDataset(Dataset):
    def __init__(self, root_dir, extensions=None, pool_num=8192, pool_type="max", length=None):
        self.root_dir = root_dir
        self.extensions = extensions if extensions else []
        self.pool_num = pool_num
        self.pool_type = pool_type
        self.samples = self._load_samples()
        self.samples_keys = sorted(self.samples.keys())
        self.length = length
        if length is not None:
            if length > len(self.samples_keys):
                pass # enlarge the dataset
            elif length > 0:
                self.samples_keys = self.samples_keys[:length]
            elif length < 0:
                self.samples_keys = self.samples_keys[length:]
            elif length == 0:
                raise ValueError("length must be a non-zero value!")
        else:
            self.length = len(self.samples_keys)

    def _load_samples(self):
        files = os.listdir(self.root_dir)
        samples = {}
        for file in files:
            file_path = os.path.join(self.root_dir, file)
            sample_id, ext = file.split(".",maxsplit=1)
            if ext in self.extensions:
                if sample_id in samples.keys():
                    samples[sample_id][ext] = file_path
                else:
                    samples[sample_id]={"subj": file_path}
                    samples[sample_id][ext] = file_path
            # print(samples)
        return samples
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image
    
    def _load_npy(self, npy_path):
        array = np.load(npy_path)
        array = torch.from_numpy(array)
        return array
    
    # def vox_process(self, x):
    #     if self.pool_num is not None:
    #         x = pool_voxels(x, self.pool_num, self.pool_type)
    #     return x
    
    def subj_process(self, key):
        id = int(key.split("/")[-2].split("subj")[-1])
        return id
    
    def aug_process(self, brain3d):
        return brain3d

    def __len__(self):
        # return len(self.samples_keys)
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.samples_keys)
        sample_key = self.samples_keys[idx]
        sample = self.samples[sample_key]
        items = []
        for ext in self.extensions:
            if ext == "jpg":
                items.append(self._load_image(sample[ext]))
            elif ext == "nsdgeneral.npy":
                voxel = self._load_npy(sample[ext])
                items.append(voxel)
                # items.append(self.vox_process(voxel))
            elif ext == "coco73k.npy":
                items.append(self._load_npy(sample[ext]))
            elif ext == "subj":
                items.append(self.subj_process(sample[ext]))
            elif ext == "wholebrain_3d.npy":
                brain3d = self._load_npy(sample[ext])
                items.append(self.aug_process(brain3d, ))

        return items


def get_dls(subject, data_path, batch_size, val_batch_size, num_workers):
    train_path = "{}/webdataset_avg_split/train/train_subj0{}".format(data_path, subject)
    val_path = "{}/webdataset_avg_split/val/val_subj0{}".format(data_path, subject)
    test_path = "{}/webdataset_avg_split/test/test_subj0{}".format(data_path, subject)
    extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy']

    train_dl = get_dataloader(
        train_path,
        batch_size=batch_size,
        num_workers=num_workers,
        extensions=extensions,
        is_shuffle=True,
    )

    val_dl = get_dataloader(
        val_path,
        batch_size=val_batch_size,
        num_workers=num_workers,
        extensions=extensions,
        is_shuffle=False,
    )

    test_dl = get_dataloader(
        test_path,
        batch_size=val_batch_size,
        num_workers=num_workers,
        is_shuffle=False,
        extensions=extensions
    )

    num_train=len(train_dl.dataset)
    num_val=len(val_dl.dataset)
    print(train_path,"\n",val_path)
    print("number of train data:", num_train)
    print("batch_size", batch_size)
    print("number of val data:", num_val)
    print("val_batch_size", val_batch_size)

    return train_dl, val_dl, test_dl


def get_dataloader(
        root_dir,
        batch_size,
        num_workers=1,
        seed=42,
        is_shuffle=True,
        extensions=['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"],
        pool_type=None,
        pool_num=None,
        length=None,
    ):
    seed_everything(seed)
    dataset = NSDDataset(root_dir=root_dir, extensions=extensions, pool_num=pool_num, pool_type=pool_type, length=length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=is_shuffle)

    return dataloader

# process the bounding boxes
def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box

def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def draw_bounding_boxes(
        image,
        boxes,
        **kwargs,
):
    if isinstance(image, Image.Image):
        image = transforms.PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    return _draw_bounding_boxes(image, boxes, **kwargs)

colors = ['#ed7d31', '#5b9bd5', '#70ad47', '#7030a0', '#c00000', '#ffff00', "olive", "brown", "cyan"]
pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

def extract_boxes(string):
    ret = []
    for bboxes_str in pat.findall(string):
        bboxes = []
        bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
        for bbox_str in bbox_strs:
            bbox = list(map(float, bbox_str.split(',')))
            bboxes.append(bbox)
        ret.append(bboxes)
    return ret
    
def postprocess(text, image, width=8):
    if image is None:
        return text, None

    image = expand2square(image)

    extract_pred = extract_boxes(text)
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(extract_pred):
        color = colors[idx % len(colors)]
        for box in boxes:
            boxes_to_draw.append(de_norm_box_xyxy(box, w=image.width, h=image.height))
            color_to_draw.append(color)
    if not boxes_to_draw:
        return text, None
    res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=width)
    res = transforms.ToPILImage()(res)

    # post process text color
    location_text = text
    edit_text = list(text)
    bboxes_str = pat.findall(text)
    for idx in range(len(bboxes_str) - 1, -1, -1):
        color = colors[idx % len(colors)]
        boxes = bboxes_str[idx]
        span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
        location_text = location_text[:span[0]]
        edit_text[span[0]:span[1]] = f'<span style="color:{color}; font-weight:bold;">{boxes}</span>'
    text = "".join(edit_text)
    return text, res


# def soft_clip_loss(preds, targs, temp=0.005, eps=1e-10):

#     clip_clip = (targs @ targs.T)/temp + eps
#     brain_clip = (preds @ targs.T)/temp + eps
    
#     loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
#     loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
#     loss = (loss1 + loss2)/2
#     return loss


def cross_modal_late_interaction(image_tokens, text_tokens):
    # import ipdb; ipdb.set_trace()
    n1, n2 = image_tokens.size(1), text_tokens.size(1)  # Number of tokens per image and text
    
    # Compute token-wise similarities
    sim_matrix = torch.einsum('bik,bjk->bij', image_tokens, text_tokens)

    # Image-to-Text similarity
    max_sim_I = torch.max(sim_matrix, dim=2)[0]  # Max similarity for each image token across text tokens
    # sim_I = torch.mean(max_sim_I, dim=1)  # Average max similarity across image tokens
    loss_I2T = -torch.mean(F.log_softmax(max_sim_I, dim=1))

    # Text-to-Image similarity
    max_sim_T = torch.max(sim_matrix, dim=1)[0]  # Max similarity for each text token across image tokens
    # sim_T = torch.mean(max_sim_T, dim=1)  # Average max similarity across text tokens
    loss_T2I = -torch.mean(F.log_softmax(max_sim_T, dim=1))

    # Combine similarities for final cross-modal similarity score
    # cross_modal_similarity = (sim_I + sim_T) / 2
    cross_modal_similarity_loss = (loss_I2T + loss_T2I) / 2

    return cross_modal_similarity_loss


import torch.nn.functional as F

def contrastive_loss_fn(anchor, positive, temperature=0.07):
    similarity = F.cosine_similarity(anchor, positive, dim=-1)
    # 计算余弦相似度
    similarity = F.cosine_similarity(anchor, positive, dim=-1)
    # 应用温度缩放
    loss = torch.mean(-torch.log(torch.clamp(similarity, min=1e-8) / temperature))
    return loss

def soft_clip_loss(preds, targs, temp=0.05, eps=1e-10):

    clip_clip = (targs @ targs.T)/temp + eps
    brain_clip = (preds @ targs.T)/temp + eps
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


def siglip_loss(preds, targs, temp=0.05, eps=1e-10):
    logits = (preds @ targs.T) / temp + eps
    # 计算相似性分数
    logits = (preds @ targs.T) / temp + eps  # [batch_size, batch_size]
    
    # 真实标签：对角线为正样本 (1)，其他为负样本 (0)
    batch_size = preds.shape[0]
    labels = torch.eye(batch_size, device=logits.device)  # 对角矩阵，1 表示正样本对
    
    # 对 logits 应用 sigmoid，转换为概率
    probs = torch.sigmoid(logits)
    
    # 计算交叉熵损失（对称形式）
    loss1 = -(labels * torch.log(probs + eps) + (1 - labels) * torch.log(1 - probs + eps)).mean()
    loss2 = -(labels.T * torch.log(probs.T + eps) + (1 - labels.T) * torch.log(1 - probs.T + eps)).mean()
    
    # 平均双向损失
    loss = (loss1 + loss2) / 2
    return loss



import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def draw_similarity_matrix(train_i, data, crop_sims_volex):
    data = data.detach().cpu().numpy()
    crop_sims_volex = crop_sims_volex.detach().cpu().numpy()
    # import ipdb; ipdb.set_trace()
    mean_value = np.mean(data)
    min_value = np.min(data)
    max_value = np.max(data)
    try:
        import seaborn as sns
    except ImportError:
        class _SeabornFallback:
            def set(self, *args, **kwargs):
                return None

        sns = _SeabornFallback()

    sns.set(style="whitegrid")  # 设置 Seaborn 风格
    plt.figure(figsize=(12, 6))  # 调整图像大小
    plt.plot(data, marker='o', linestyle='-', markersize=4, alpha=0.8, label="Data Trend")

    # 添加水平参考线
    plt.axhline(mean_value, color='r', linestyle='--', linewidth=1.2, label=f"Mean: {mean_value:.4f}")
    plt.axhline(min_value, color='g', linestyle='--', linewidth=1.2, label=f"Min: {min_value:.4f}")
    plt.axhline(max_value, color='b', linestyle='--', linewidth=1.2, label=f"Max: {max_value:.4f}")
    plt.axhline(crop_sims_volex, color='purple', linestyle='-.', linewidth=1.5, label=f"Crop Sims: {crop_sims_volex:.4f}")

    # 图像标题和标签
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Similarity Matrix Analysis", fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=10)  # 设置图例字体大小
    plt.grid(True, linestyle='--', alpha=0.6)  # 轻量级网格
    plt.tight_layout()  # 解决布局溢出问题

    # 保存图片
    plt.savefig(f'./sim_matrix_figs_5/sim_matrix_{train_i}.png')
    plt.close()  # 关闭图像，防止重复绘制





# def volex_to_image_cross_modal_late_interaction(volex_tokens, image_tokens):
#     n1, n2 = image_tokens.size(1), volex_tokens.size(1)  # Number of tokens per image and text
    
#     volex_tokens = F.normalize(volex_tokens, p=2, dim=-1)  # 归一化到单位向量
#     image_tokens = F.normalize(image_tokens, p=2, dim=-1)  # 归一化到单位向量

#     # Compute token-wise similarities
#     sim_matrix = torch.einsum('bik,bjk->bij', volex_tokens, image_tokens)
#     # print(sim_matrix.max(), sim_matrix.min())
    
#     sims_volex_embed_to_image_embddings = torch.sum(sim_matrix, dim=2) / n2  # 形状：[batch_size, n1]
#     # import ipdb; ipdb.set_trace()
#     max_sim_idx = torch.argmax(sims_volex_embed_to_image_embddings, dim=1)  # 形状：[batch_size]
#     # print(torch.argmax(sims_volex_embed_to_image_embddings, dim=1))

#     selected_volex_tokens = volex_tokens[torch.arange(volex_tokens.size(0)), max_sim_idx]

#     image_embedding_avg = torch.mean(image_tokens, dim=1)

#     # import ipdb; ipdb.set_trace()
#     contrastive_loss = soft_clip_loss(selected_volex_tokens, image_embedding_avg)
    

#     return contrastive_loss
import torch.nn as nn
def volex_to_image_cross_modal_late_interaction(train_i, volex_tokens, image_tokens, max_cat_ids, memory_bank, salient_category, loss_fn, topk_k=5):

    n1, n2= image_tokens.size(1), volex_tokens.size(1)  # Number of tokens per image and volex
    
    volex_tokens = F.normalize(volex_tokens, p=2, dim=-1)  # 归一化到单位向量
    image_tokens = F.normalize(image_tokens, p=2, dim=-1)  # 归一化到单位向量
    # text_tokens = F.normalize(text_tokens, p=2, dim=-1)  # 归一化到单位向量

    # crop_sims_volex = nn.functional.cosine_similarity(volex_tokens[0], image_tokens[0]).mean()

    sim_matrix_img = torch.einsum('bik,bjk->bij', volex_tokens, image_tokens)
    # sim_matrix_txt = torch.einsum('bik,bjk->bij', volex_tokens, text_tokens)

    sims_volex_embed_to_image_embddings = torch.sum(sim_matrix_img, dim=2) / n2  # 形状：[batch_size, n1]

    # draw_similarity_matrix(train_i, sims_volex_embed_to_image_embddings[0], crop_sims_volex)

    value, index = torch.topk(sims_volex_embed_to_image_embddings, k=topk_k, dim=1)  # 形状：[batch_size, topk_k]

    # import ipdb; ipdb.set_trace()
    selected_volex_tokens = torch.gather(
    volex_tokens,
    dim=1,
    index=index.unsqueeze(-1).expand(-1, -1, volex_tokens.size(-1))) # shape: [batch_size, topk_k, 1024]

    image_embedding_avg = torch.mean(image_tokens, dim=1) # (B 1024)
    # text_embedding_avg = torch.mean(text_tokens, dim=1) # (B 1024)
    selected_volex_tokens_avg = torch.mean(selected_volex_tokens, dim=1) # (B 1024)

    contrastive_loss_img = soft_clip_loss(selected_volex_tokens_avg, image_embedding_avg, temp=0.5)

    # contrastive_loss_text = soft_clip_loss(selected_volex_tokens_avg, text_embedding_avg)
    # salient_features_selected_tokens = []
    # for i,id in enumerate(max_cat_ids):
    #     if id in salient_category:
    #         # memory_bank.enqueue(id, selected_volex_tokens_avg[i:i+1,:].detach())
    #         memory_bank.enqueue(id, image_embedding_avg[i])
    #         salient_features_selected_tokens.append(selected_volex_tokens_avg[i])
    # salient_features_tokens = torch.stack(salient_features_selected_tokens, dim=0)
    
    # sim_features = []

    # for id in max_cat_ids:
    #     if id in salient_category:
    #         sim_features.append(memory_bank.get_memory_bank(id))
    # sim_features = torch.stack(sim_features, dim=0)
    # # import ipdb; ipdb.set_trace()


    # return contrastive_loss_img, contrastive_loss_aux
    return contrastive_loss_img



def volex_to_image_cross_modal_late_interaction_all_align(train_i, volex_tokens, image_tokens, max_cat_ids, memory_bank, salient_category, loss_fn, topk_k=5):

    clip_image_pred_norm = nn.functional.normalize(volex_tokens.flatten(1), dim=-1)
    clip_image_norm = nn.functional.normalize(image_tokens.flatten(1), dim=-1)

    # contrastive_loss_img = soft_clip_loss(volex_tokens, image_tokens, temp=0.005)
    contrastive_loss_img = soft_clip_loss(
        clip_image_pred_norm,
        clip_image_norm,
        temp = 0.005
        )

    
    mse_loss_img = loss_fn(clip_image_pred_norm, clip_image_norm) * 200000

    return contrastive_loss_img, mse_loss_img




def volex_to_image_cross_modal_late_interaction_avg_topk(volex_tokens, image_tokens, max_cat_ids, memory_bank, salient_category, loss_fn, topk_k=10):

    n1, n2= image_tokens.size(1), volex_tokens.size(1)  # Number of tokens per image and volex
    
    volex_tokens = F.normalize(volex_tokens, p=2, dim=-1)  # 归一化到单位向量
    image_tokens = F.normalize(image_tokens, p=2, dim=-1)  # 归一化到单位向量
    # text_tokens = F.normalize(text_tokens, p=2, dim=-1)  # 归一化到单位向量

    sim_matrix_img = torch.einsum('bik,bjk->bij', volex_tokens, image_tokens)
    # sim_matrix_txt = torch.einsum('bik,bjk->bij', volex_tokens, text_tokens)

    sims_volex_embed_to_image_embddings = torch.sum(sim_matrix_img, dim=2) / n2  # 形状：[batch_size, n1]

    # print('sims_volex_embed_to_image_embddings[0]', sims_volex_embed_to_image_embddings[0])
    # print('Min sims_volex_embed_to_image_embddings[0]', sims_volex_embed_to_image_embddings[0].min())
    # print('Max sims_volex_embed_to_image_embddings[0]', sims_volex_embed_to_image_embddings[0].max())


    avg_sim = torch.mean(sims_volex_embed_to_image_embddings, dim=1, keepdim=True)  # 形状：[batch_size, 1]

    # print('avg_sim[0]', avg_sim[0])

    mask = sims_volex_embed_to_image_embddings > avg_sim  # 形状：[batch_size, n1]

    # 使用 masked_select 获取符合条件的 tokens 索引
    batch_size = volex_tokens.size(0)
    feature_dim = volex_tokens.size(-1)

    selected_volex_tokens_avg = torch.zeros(batch_size, feature_dim, device=volex_tokens.device)  # Shape: [batch_size, feature_dim]

    for i in range(batch_size):
        selected_indices = torch.where(mask[i])[0]
        if len(selected_indices) > 0:
            selected_tokens = volex_tokens[i].index_select(0, selected_indices)
            selected_volex_tokens_avg[i] = torch.mean(selected_tokens, dim=0)
        else:
            # 如果没有 image token 大于平均值，取相似度最大的 token
            selected_volex_tokens_avg[i] = torch.mean(volex_tokens[i], dim=0)  # Fallback to full average


    # import ipdb; ipdb.set_trace()


    # selected_volex_tokens = torch.gather(
    # volex_tokens,
    # dim=1,
    # index=index.unsqueeze(-1).expand(-1, -1, volex_tokens.size(-1))) # shape: [batch_size, topk_k, 1024]

    image_embedding_avg = torch.mean(image_tokens, dim=1)  # Shape: [batch_size, feature_dim]
    # text_embedding_avg = torch.mean(text_tokens, dim=1) # (B 1024)
    # selected_volex_tokens_avg = torch.mean(selected_volex_tokens, dim=1) # (B 1024)

    contrastive_loss_img = soft_clip_loss(selected_volex_tokens_avg, image_embedding_avg, temp=0.005)
    mse_loss_img = loss_fn(selected_volex_tokens_avg, image_embedding_avg) * 200000
    # contrastive_loss_text = soft_clip_loss(selected_volex_tokens_avg, text_embedding_avg)
    salient_features_selected_tokens = []
    for i,id in enumerate(max_cat_ids):
        if id in salient_category:
            # memory_bank.enqueue(id, selected_volex_tokens_avg[i:i+1,:].detach())
            memory_bank.enqueue(id, image_embedding_avg[i])
            salient_features_selected_tokens.append(selected_volex_tokens_avg[i])
    salient_features_tokens = torch.stack(salient_features_selected_tokens, dim=0)
    
    sim_features = []

    for id in max_cat_ids:
        if id in salient_category:
            sim_features.append(memory_bank.get_memory_bank(id))
    sim_features = torch.stack(sim_features, dim=0)
    # import ipdb; ipdb.set_trace()

    sim_features = sim_features.to(salient_features_tokens.device)
    contrastive_loss_aux = soft_clip_loss(sim_features, salient_features_tokens, temp=0.005)
    mse_loss_aux = loss_fn(sim_features, salient_features_tokens) * 200000

    return contrastive_loss_img, contrastive_loss_aux, mse_loss_img, mse_loss_aux



def volex_to_image_cross_modal_late_interaction_old(volex_tokens, clip_tokens, topk_k=1):
    n1, n2 = clip_tokens.size(1), volex_tokens.size(1)  # Number of tokens per image and volex
    # import ipdb; ipdb.set_trace()
    volex_tokens = F.normalize(volex_tokens, p=2, dim=-1)  # 归一化到单位向量
    clip_tokens = F.normalize(clip_tokens, p=2, dim=-1)  # 归一化到单位向量

    # Compute token-wise similarities
    sim_matrix = torch.einsum('bik,bjk->bij', volex_tokens, clip_tokens)
    # print(sim_matrix.max(), sim_matrix.min())
    
    sims_volex_embed_to_image_embddings = torch.sum(sim_matrix, dim=2) / n2  # 形状：[batch_size, n1]
    # print(f'Max: {sims_volex_embed_to_image_embddings[0].max()}, Min: {sims_volex_embed_to_image_embddings[0].min()}')
    value, index = torch.topk(sims_volex_embed_to_image_embddings, k=topk_k, dim=1)  # 形状：[batch_size, topk_k]

    selected_volex_tokens = torch.gather(
        volex_tokens,
        dim=1,
        index=index.unsqueeze(-1).expand(-1, -1, volex_tokens.size(-1))
    ) # shape: [batch_size, topk_k, 1024]
    # import ipdb; ipdb.set_trace()
    image_embedding_avg = torch.mean(clip_tokens, dim=1) # (B 1024)
    selected_volex_tokens_avg = torch.mean(selected_volex_tokens, dim=1) # (B 1024)
 
    contrastive_loss = soft_clip_loss(selected_volex_tokens_avg, image_embedding_avg, temp=0.5)
    
    return contrastive_loss


def crop_tensor_image(tensor_image, bbox):
    """
    从 Tensor 图像裁剪指定的区域。

    参数:
    tensor_image: Tensor (C, H, W)
    bbox: tuple (x_min, y_min, width, height)

    返回:
    裁剪后的 Tensor
    """
    x_min, y_min, width, height = bbox
    # 确保索引为整数（可以用 round, floor, 或 ceil）
    x_min = int(round(x_min))
    y_min = int(round(y_min))
    x_max = int(round(x_min + width))
    y_max = int(round(y_min + height))
    return tensor_image[:, y_min:y_max, x_min:x_max]



def calculate_transformed_bbox(original_size, bbox, target_size=256):

    original_width, original_height = original_size
    scale = target_size / min(original_width, original_height)
    
    # 缩放后的bbox
    new_bbox_xmin = bbox[0] * scale
    new_bbox_ymin = bbox[1] * scale
    new_bbox_xmax = bbox[2] * scale
    new_bbox_ymax = bbox[3] * scale

    # 缩放后的图像宽高
    new_width = original_width * scale
    new_height = original_height * scale

    # 计算CenterCrop偏移量
    crop_x_offset = (new_width - target_size) / 2
    crop_y_offset = (new_height - target_size) / 2

    # # 调整bbox
    final_bbox_xmin = max(new_bbox_xmin - crop_x_offset, 0)
    final_bbox_ymin = max(new_bbox_ymin - crop_y_offset, 0)
    final_bbox_xmax = min(new_bbox_xmin + new_bbox_xmax - crop_x_offset, target_size) - max(new_bbox_xmin - crop_x_offset, 0)
    final_bbox_ymax = min(new_bbox_ymin + new_bbox_ymax - crop_y_offset, target_size) - max(new_bbox_ymin - crop_y_offset, 0)

    
    return (final_bbox_xmin, final_bbox_ymin, final_bbox_xmax, final_bbox_ymax)


def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

# from utils.nsd_access import NSDAccess
def prepare_coco(nsda):
    # Preload coco captions
    # nsda = NSDAccess(args.data_path)
    coco_73k = list(range(0, NSD_NUM_IMAGES))
    prompts_list = nsda.read_image_coco_info(coco_73k,info_type='captions')

    print("coco captions loaded.")

    return prompts_list
