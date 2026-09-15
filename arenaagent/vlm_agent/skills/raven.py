import subprocess
from typing import Dict, List, Optional
import os
import glob
from PIL import Image
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SKILLS_DIR, "Resnet18_MLP_epoch_199.pth")
EMBEDDING_PATH = os.path.join(SKILLS_DIR, "embedding.npy")
DEFAULT_STRUCTURE = ['Scene', 'Singleton', 'Grid', 'Center_Single', '/', '/', '/', '/']
_MODEL_CACHE = None
_EMBEDDINGS_CACHE = None

class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        # self.file_names = ["/home/zsy/VLAAgent/assets/Raven/reconstructed.npz"]
        self.img_size = img_size
        self.embeddings = np.load(EMBEDDING_PATH, allow_pickle=True)
        self.shuffle = shuffle

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        structure = ['Scene', 'Singleton', 'Grid', 'Center_Single', '/', '/', '/', '/']
        # structure = data["structure"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        resize_image = []
        for idx in range(0, 16):
            # resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
            img_pil = Image.fromarray(image[idx,:,:].astype(np.uint8))
            img_resized = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            resize_image.append(np.array(img_resized))
        resize_image = np.stack(resize_image)
        # image = resize(image, (16, 128, 128))
        # meta_matrix = data["mata_matrix"]

        embedding = torch.zeros((6, 300), dtype=torch.float)
        indicator = torch.zeros(1, dtype=torch.float)
        element_idx = 0
        for element in structure:
            # print("element: ", element)
            if element != '/':
                embedding[element_idx, :] = torch.tensor(self.embeddings.item().get(element), dtype=torch.float)
                element_idx += 1
        if element_idx == 6:
            indicator[0] = 1.
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            # meta_matrix = self.transform(meta_matrix)
            target = torch.tensor(target, dtype=torch.long)
        return resize_image, target, embedding, indicator
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.model
    
    def load_model(self, path, epoch):
        # state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch))['state_dict']
        if torch.cuda.is_available():
            state_dict = torch.load(path, weights_only=False)['state_dict']
        else:
            state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)['state_dict']
        self.load_state_dict(state_dict)

    def save_model(self, path, epoch, acc, loss):
        torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, path+'{}_epoch_{}.pth'.format(self.name, epoch))

    def test_(self, image, target, embedding, indicator):
        with torch.no_grad():
            output = self(image, embedding, indicator)
        # pred = output[0].data.max(1)[1]
        probs = torch.softmax(output[0], dim=1)
        return probs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCTreeNet(torch.nn.Module):
    def __init__(self, in_dim=300, img_dim=256, use_cuda=True):
        '''
        initialization for TreeNet model, basically a ChildSumLSTM model
        with non-linear activation embedding for different nodes in the AoG.
        Shared weigths for all LSTM cells.
        :param in_dim:      input feature dimension for word embedding (from string to vector space)
        :param img_dim:     dimension of the input image feature, should be (panel_pair_number * img_feature_dim (e.g. 512 or 256))
        '''
        super(FCTreeNet, self).__init__()
        self.in_dim = in_dim
        self.img_dim = img_dim
        self.fc = nn.Linear(self.in_dim, self.in_dim)
        self.leaf = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.middle = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.merge = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.root = nn.Linear(self.in_dim + self.img_dim, self.img_dim)

        self.relu = nn.ReLU()

    def forward(self, image_feature, input, indicator):
        '''
        Forward funciton for TreeNet model
        :param input:		input should be (batch_size * 6 * input_word_embedding_dimension), got from the embedding vector
        :param indicator:	indicating whether the input is of structure with branches (batch_size * 1)
        :param image_feature:   input dictionary for each node, primarily feature, for example (batch_size * 16 (panel_pair_number) * feature_dim (output from CNN))
        :return:
        '''
        # image_feature = image_feature.view(-1, 16, image_feature.size(2))
        input = self.fc(input.view(-1, input.size(-1)))
        input = input.view(-1, 6, input.size(-1))
        input = input.unsqueeze(1).repeat(1, image_feature.size(1), 1, 1)
        indicator = indicator.unsqueeze(1).repeat(1, image_feature.size(1), 1).view(-1, 1)

        leaf_left = input[:, :, 3, :].view(-1, input.size(-1))           # (batch_size * panel_pair_num) * input_word_embedding_dimension
        leaf_right = input[:, :, 5, :].view(-1, input.size(-1))
        inter_left = input[:, :, 2, :].view(-1, input.size(-1))
        inter_right = input[:, :, 4, :].view(-1, input.size(-1))
        merge = input[:, :, 1, :].view(-1, input.size(-1))
        root = input[:, :, 0, :].view(-1, input.size(-1))
        
        # concating image_feature and word_embeddings for leaf node inputs
        leaf_left = torch.cat((leaf_left, image_feature.view(-1, image_feature.size(-1))), dim=-1)
        leaf_right = torch.cat((leaf_right, image_feature.view(-1, image_feature.size(-1))), dim=-1)

        out_leaf_left = self.leaf(leaf_left)
        out_leaf_right = self.leaf(leaf_right)

        out_leaf_left = self.relu(out_leaf_left)
        out_leaf_right = self.relu(out_leaf_right)

        out_left = self.middle(torch.cat((inter_left, out_leaf_left), dim=-1))
        out_right = self.middle(torch.cat((inter_right, out_leaf_right), dim=-1))

        out_left = self.relu(out_left)
        out_right = self.relu(out_right)

        out_right = torch.mul(out_right, indicator)
        merge_input = torch.cat((merge, out_left + out_right), dim=-1)
        out_merge = self.merge(merge_input)

        out_merge = self.relu(out_merge)

        out_root = self.root(torch.cat((root, out_merge), dim=-1))
        out_root = self.relu(out_root)
        # size ((batch_size * panel_pair) * feature_dim)
        return out_root
    
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
    
    def forward(self, x):
        return x

class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8+9+21)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Resnet18_MLP(BasicModel):
    def __init__(self, args):
        super(Resnet18_MLP, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.mlp = mlp_module()
        self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred, meta_target_pred, meta_struct_pred = output[0], output[1], output[2]

        target_loss = F.cross_entropy(pred, target)
        meta_target_pred = torch.chunk(meta_target_pred, chunks=9, dim=1)
        meta_target = torch.chunk(meta_target, chunks=9, dim=1)
        meta_target_loss = 0.
        for idx in range(0, 9):
            meta_target_loss += F.binary_cross_entropy(F.sigmoid(meta_target_pred[idx]), meta_target[idx])

        meta_struct_pred = torch.chunk(meta_struct_pred, chunks=21, dim=1)
        meta_structure = torch.chunk(meta_structure, chunks=21, dim=1)
        meta_struct_loss = 0.
        for idx in range(0, 21):
            meta_struct_loss += F.binary_cross_entropy(F.sigmoid(meta_struct_pred[idx]), meta_structure[idx])
        loss = target_loss + self.meta_alpha*meta_struct_loss/21. + self.meta_beta*meta_target_loss/9.
        return loss

    def forward(self, x, embedding, indicator):
        alpha = 1.0
        features = self.resnet18(x.view(-1, 16, 224, 224))
        features_tree = features.view(-1, 1, 512)
        features_tree = self.fc_tree_net(features_tree, embedding, indicator)
        final_features = features + alpha * features_tree
        output = self.mlp(final_features)
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred


import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# 坐标文件：与本脚本同级目录，与 show_raven_npz 写入路径一致
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COORDS_JSON_FILENAME = "group_coords.json"
COORDS_JSON_PATH = os.path.join(_SCRIPT_DIR, COORDS_JSON_FILENAME)

# 为了去除可见黑色边框，在每个坐标四周再内缩的像素数
BORDER_PIXELS = 4


def crop_group_image_to_pil_groups(
    image_or_path,
    output_size: tuple = (400, 400),
):
    """Crop a competition canvas into three in-memory 16-panel groups."""
    if not os.path.exists(COORDS_JSON_PATH):
        raise FileNotFoundError(
            f"未找到坐标文件 {COORDS_JSON_PATH}，请先用 show_raven_npz 生成大图并写入坐标。"
        )

    with open(COORDS_JSON_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("RGB")
    else:
        img = Image.open(image_or_path).convert("RGB")
    w_img, h_img = img.size
    fig_info = meta.get("figure", {})
    w_fig = fig_info.get("width_px", w_img)
    h_fig = fig_info.get("height_px", h_img)
    sx = w_img / float(w_fig) if w_fig else 1.0
    sy = h_img / float(h_fig) if h_fig else 1.0

    groups = []
    for q_idx, question in enumerate(meta.get("questions", [])[:3]):
        group = []
        for kind in ("problem", "answer"):
            boxes = question.get(kind, [])[:8]
            if len(boxes) != 8:
                raise ValueError(f"question {q_idx + 1} has {len(boxes)} {kind} boxes, expected 8")
            for box in boxes:
                x0 = int(round(box["x0"] * sx))
                y0 = int(round(box["y0"] * sy))
                x1 = int(round(box["x1"] * sx))
                y1 = int(round(box["y1"] * sy))
                x0 += BORDER_PIXELS * 2 + (1 if kind == "answer" else 0)
                y0 += BORDER_PIXELS
                x1 -= BORDER_PIXELS * 2
                y1 -= BORDER_PIXELS
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w_img, x1), min(h_img, y1)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError(f"invalid crop box for question {q_idx + 1} {kind}")
                cropped = img.crop((x0, y0, x1, y1))
                group.append(cropped.resize(output_size, Image.Resampling.LANCZOS))
        groups.append(group)

    if len(groups) != 3:
        raise ValueError(f"canvas has {len(groups)} questions, expected 3")
    return groups


def crop_group_image_to_subplots(
    image_path: str,
    output_size: tuple = (400, 400),
):
    """
    读取同级目录的 group_coords.json，按坐标从大图中裁剪出 16*3=48 张子图并保存。

    Args:
        image_path: 大图路径（如 group_01.png）
        output_size: 每个裁剪子图 resize 到的 (宽, 高)，默认 (400, 400)

    Raises:
        FileNotFoundError: 当 group_coords.json 不存在时
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    base_dir = os.path.dirname(image_path)
    subfolder = os.path.join(base_dir, base_name)
    os.makedirs(subfolder, exist_ok=True)
    groups = crop_group_image_to_pil_groups(image_path, output_size=output_size)
    saved = 0
    for q_idx, group in enumerate(groups):
        col_prefix = f"question_{q_idx+1}"
        for idx, cropped in enumerate(group[:8]):
            out_path = os.path.join(subfolder, f"{col_prefix}_problem_{idx+1:02d}.png")
            cropped.save(out_path)
            saved += 1
        for idx, cropped in enumerate(group[8:]):
            out_path = os.path.join(subfolder, f"{col_prefix}_answer_{idx+1:02d}.png")
            cropped.save(out_path)
            saved += 1

    print(
        f"已基于坐标从 {os.path.basename(image_path)} 裁剪出 {saved} 张子图，保存到: {subfolder}"
    )

def solve_raven(image_list, structure=[]):
    parser = argparse.ArgumentParser(description='our_model')
    parser.add_argument('--model', type=str, default='Resnet18_MLP')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--load_workers', type=int, default=16)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--path', type=str, default='/home/zsy/RAVEN/assets/')
    parser.add_argument('--save', type=str, default=MODEL_PATH)
    parser.add_argument('--structure', type=list, default=[])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--meta_alpha', type=float, default=0.0)
    parser.add_argument('--meta_beta', type=float, default=0.0)


    args = parser.parse_args()
    global _MODEL_CACHE, _EMBEDDINGS_CACHE
    if _MODEL_CACHE is None:
        model = Resnet18_MLP(args)
        model_path = args.save if os.path.isabs(args.save) else os.path.join(SKILLS_DIR, args.save)
        model.load_model(model_path, 0)
        model.eval()
        _MODEL_CACHE = model
    model = _MODEL_CACHE
    if not structure:
        # The bundled checkpoint was trained with this RAVEN A-SIG structure.
        # Supplying it restores the structural reasoning branch at inference.
        structure = DEFAULT_STRUCTURE

    probs_list = []
    for image in image_list:
        image_array = []
        for img in image:
            # img = img.resize((224, 224))
            image_array.append(np.array(img))

        image_array = np.stack(image_array)

        resize_image = []
        for idx in range(0, 16):
            # resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
            img_pil = Image.fromarray(image_array[idx,:,:].astype(np.uint8))
            img_resized = img_pil.resize((args.img_size, args.img_size), Image.BILINEAR)
            resize_image.append(np.array(img_resized))
        resize_image = np.stack(resize_image)
        transform = transforms.Compose([ToTensor()])
        resize_image = transform(resize_image)

        image = resize_image

        accuracy = 0

        acc_all = 0.0
        counter = 0
        target = 5

        embedding = torch.zeros((6, 300), dtype=torch.float)
        indicator = torch.zeros(1, dtype=torch.float)
        element_idx = 0
        # embeddings = np.load("embedding.npy", allow_pickle=True)
        if _EMBEDDINGS_CACHE is None:
            try:
                # Python2-pickled npy files need latin1 encoding when loaded in py3
                _EMBEDDINGS_CACHE = np.load(EMBEDDING_PATH, allow_pickle=True, encoding='latin1')
            except TypeError:
                # Fallback for older numpy versions without encoding arg
                _EMBEDDINGS_CACHE = np.load(EMBEDDING_PATH, allow_pickle=True)
        embeddings = _EMBEDDINGS_CACHE
        for element in structure:
            if element != '/':
                embedding[element_idx, :] = torch.tensor(embeddings.item().get(element), dtype=torch.float)
                element_idx += 1
        if element_idx == 6:
            indicator[0] = 1.
        probs = model.test_(image, target, embedding, indicator)
        probs_list.append(probs.reshape(8))

    idx = 0
    p1 = probs_list[idx]
    p2 = probs_list[idx + 1]
    p3 = probs_list[idx + 2]

    # 枚举所有三元组组合
    triple_probs = []
    triple_labels = []

    for c1 in range(8):
        for c2 in range(8):
            for c3 in range(8):
                triple_probs.append(p1[c1] * p2[c2] * p3[c3])
                triple_labels.append((c1+1, c2+1, c3+1))  #mapping 0-7 to 1-8

    triple_probs = torch.stack(triple_probs)  # (C^3,)
    triple_labels = torch.tensor(triple_labels)  # (C^3, 3)

    # 按概率排序
    sorted_idx = torch.argsort(triple_probs, descending=True)
    sorted_triples = triple_labels[sorted_idx]
    return sorted_triples.tolist()


if __name__ == '__main__':
    json_path = ""

    # with open(json_path) as f:
    #     data = json.load(f)

    image_paths = "group_01.png"
    img_list_0 = []

    group_01_path = "group_01.png"
    if os.path.exists(group_01_path):
        crop_group_image_to_subplots(group_01_path)
    else:
        print(f"图片不存在: {group_01_path}")

    for i in range(3):
        image_list = []
        for n in range(1, 9):
            problem_name = f"question_{i+1}_problem_0{n}.png".format(n)
            problem_path = os.path.join("group_01/", problem_name)
            if not os.path.exists(problem_path):
                raise IOError("Missing file: {}".format(problem_path))
            image_list.append(Image.open(problem_path).convert("L"))

        for n in range(1, 9):
            answer_name = f"question_{i+1}_answer_0{n}.png".format(n)
            answer_path = os.path.join("group_01/", answer_name)
            if not os.path.exists(answer_path):
                raise IOError("Missing file: {}".format(problem_path))
            image_list.append(Image.open(answer_path).convert("L"))
        
        img_list_0.append(image_list)
    
    pred = solve_raven(img_list_0)

    output = {
        "success": True,
        "result": pred
    }

    print(output)
