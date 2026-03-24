import numpy as np
import torch


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_irsu(batch):
    items_t, ratings_t, sizes_t, users_t = (
        batch["items"],
        batch["ratings"],
        batch["sizes"],
        batch["users"],
    )
    return items_t, ratings_t, sizes_t, users_t


def batch_no_embeddings(batch, frame_size, *args, **kwargs):
    """
    Embed Batch: discrete state discrete action
    """
    items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
    b_size = ratings_t.size(0)
    items = items_t[:, :-1]
    next_items = items_t[:, 1:]
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]
    action = items_t[:, -1]
    reward = ratings_t[:, -1]
    done = torch.zeros(b_size)

    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1
    batch = {
        "items": items,
        "next_items": next_items,
        "ratings": "ratings",
        "next_ratings": next_ratings,
        "action": action,
        "reward": reward,
        "done": done,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch


def batch_tensor_embeddings(batch, item_embeddings_tensor, frame_size, *args, **kwargs):
    """
    Embed Batch: continuous state continuous action
    """

    items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
    # 确保索引在有效范围内
    max_index = item_embeddings_tensor.size(0) - 1
    items_t = torch.clamp(items_t, 0, max_index)
    items_emb = item_embeddings_tensor[items_t.long()]
    b_size = ratings_t.size(0)

    items = items_emb[:, :-1, :].view(b_size, -1)
    next_items = items_emb[:, 1:, :].view(b_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]

    state = torch.cat([items, ratings], 1)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_emb[:, -1, :]
    reward = ratings_t[:, -1]

    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1

    batch = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch


def batch_contstate_discaction(
    batch, item_embeddings_tensor, frame_size, num_items, *args, **kwargs
):

    """
    Embed Batch: continuous state discrete action
    """

    items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
    items_emb = item_embeddings_tensor[items_t.long()]
    b_size = ratings_t.size(0)

    items = items_emb[:, :-1, :].view(b_size, -1)
    next_items = items_emb[:, 1:, :].view(b_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]

    state = torch.cat([items, ratings], 1)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_t[:, -1]
    reward = ratings_t[:, -1]

    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1

    one_hot_action = torch.zeros(b_size, num_items)
    one_hot_action.scatter_(1, action.view(-1, 1), 1)

    batch = {
        "state": state,
        "action": one_hot_action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch


def padder(x):
    items_t = []
    ratings_t = []
    sizes_t = []
    users_t = []
    for i in range(len(x)):
        items_t.append(torch.tensor(x[i]["items"]))
        ratings_t.append(torch.tensor(x[i]["rates"]))
        sizes_t.append(x[i]["sizes"])
        users_t.append(x[i]["users"])
    items_t = torch.nn.utils.rnn.pad_sequence(items_t, batch_first=True).long()
    ratings_t = torch.nn.utils.rnn.pad_sequence(ratings_t, batch_first=True).float()
    sizes_t = torch.tensor(sizes_t).float()
    return {"items": items_t, "ratings": ratings_t, "sizes": sizes_t, "users": users_t}


def prepare_batch_dynamic_size(batch, item_embeddings_tensor, embed_batch=None):
    item_idx, ratings_t, sizes_t, users_t = get_irsu(batch)
    item_t = item_embeddings_tensor[item_idx]
    batch = {"items": item_t, "users": users_t, "ratings": ratings_t, "sizes": sizes_t}
    return batch


def prepare_batch_static_size(
    batch, item_embeddings_tensor, frame_size=10, embed_batch=batch_tensor_embeddings
):
    item_t, ratings_t, sizes_t, users_t = [], [], [], []
    for i in range(len(batch)):
        item_t.append(batch[i]["items"])
        ratings_t.append(batch[i]["rates"])
        sizes_t.append(batch[i]["sizes"])
        users_t.append(batch[i]["users"])

    # 处理项目和评分
    item_windows = []
    rating_windows = []
    expanded_users = []
    
    for i in range(len(batch)):
        items = batch[i]["items"]
        rates = batch[i]["rates"]
        user_id = batch[i]["users"]
        
        # 生成滑动窗口
        item_window = rolling_window(items, frame_size + 1)
        rating_window = rolling_window(rates, frame_size + 1)
        
        item_windows.append(item_window)
        rating_windows.append(rating_window)
        
        # 为每个窗口复制用户ID
        expanded_users.extend([user_id] * len(item_window))

    item_t = np.concatenate(item_windows, 0)
    ratings_t = np.concatenate(rating_windows, 0)

    item_t = torch.tensor(item_t)
    users_t = torch.tensor(expanded_users)
    ratings_t = torch.tensor(ratings_t).float()
    sizes_t = torch.tensor(sizes_t)

    batch = {"items": item_t, "users": users_t, "ratings": ratings_t, "sizes": sizes_t}

    return embed_batch(
        batch=batch,
        item_embeddings_tensor=item_embeddings_tensor,
        frame_size=frame_size,
    )


def make_items_tensor(items_embeddings_key_dict):
    keys = list(sorted(items_embeddings_key_dict.keys()))
    key_to_id = dict(zip(keys, range(len(keys))))
    id_to_key = dict(zip(range(len(keys)), keys))

    items_embeddings_id_dict = {}
    for k in items_embeddings_key_dict.keys():
        items_embeddings_id_dict[key_to_id[k]] = items_embeddings_key_dict[k]
    items_embeddings_tensor = torch.stack(
        [items_embeddings_id_dict[i] for i in range(len(items_embeddings_id_dict))]
    )
    return items_embeddings_tensor, key_to_id, id_to_key


class ReplayBuffer:
    def __init__(self, buffer_size, layout):
        self.buffer = None
        self.idx = 0
        self.size = buffer_size
        self.layout = layout
        self.meta = {"step": []}
        self.flush()

    def flush(self):
        del self.buffer
        self.buffer = [torch.zeros(i) for i in self.layout]
        self.idx = 0
        self.meta["step"] = []

    def append(self, batch):
        state, action, reward, next_state, step = (
            batch["state"],
            batch["action"],
            batch["reward"],
            batch["next_state"],
            batch["step"],
        )
        self.meta["step"].append(step)
        lower = self.idx
        upper = state.size(0) + lower
        self.buffer[0][lower:upper] = state
        self.buffer[1][lower:upper] = action
        self.buffer[2][lower:upper] = reward
        self.buffer[3][lower:upper] = next_state
        self.idx = upper

    def get(self):
        state, action, reward, next_state = self.buffer
        batch = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "meta": self.meta,
        }
        return batch

    def len(self):
        return self.idx


def get_base_batch(batch, device=torch.device("cuda"), done=True, return_user_ids=False):
    # 提取基本组件
    state = batch["state"]
    action = batch["action"]
    reward = batch["reward"].unsqueeze(1)
    next_state = batch["next_state"]
    
    # 处理done
    if done and "done" in batch:
        done_tensor = batch["done"].unsqueeze(1)
    else:
        done_tensor = torch.zeros_like(reward)
    
    # 移动到设备
    state = state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    done_tensor = done_tensor.to(device)
    
    if return_user_ids:
        # 添加用户ID
        if "meta" in batch and "users" in batch["meta"]:
            user_ids = batch["meta"]["users"]
            # 确保user_ids是张量
            if not isinstance(user_ids, torch.Tensor):
                user_ids = torch.tensor(user_ids, device=device, dtype=torch.long)
            else:
                user_ids = user_ids.to(device)
        else:
            # 如果没有用户ID，使用默认值
            user_ids = torch.zeros(state.size(0), device=device, dtype=torch.long)
        return state, action, reward, next_state, user_ids
    else:
        return state, action, reward, next_state, done_tensor
