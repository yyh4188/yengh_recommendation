import torch
import numpy as np
from scipy.spatial import distance


def calculate_ndcg(relevance, k=10):
    """计算NDCG@K
    
    :param relevance: 相关性列表，按推荐顺序排列
    :param k: 推荐数量
    :return: NDCG@K值
    """
    # 计算DCG
    dcg = 0.0
    for i, rel in enumerate(relevance[:k]):
        dcg += rel / np.log2(i + 2)  # 位置从1开始
    
    # 计算IDCG（理想DCG）
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance[:k]):
        idcg += rel / np.log2(i + 2)
    
    # 防止除零
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_precision_at_k(relevance, k=10):
    """计算Precision@K
    
    :param relevance: 相关性列表，按推荐顺序排列
    :param k: 推荐数量
    :return: Precision@K值
    """
    if k == 0:
        return 0.0
    
    relevant_count = sum(rel > 0 for rel in relevance[:k])
    return relevant_count / k


def evaluate_recommendations(action, embeddings_dict, movie_dict, user_history, k=10, metric='cosine'):
    """评估推荐质量
    
    :param action: 动作向量
    :param embeddings_dict: 电影嵌入字典
    :param movie_dict: 电影名称字典
    :param user_history: 用户历史记录
    :param k: 推荐数量
    :param metric: 距离度量方式
    :return: 评估结果字典
    """
    # 距离函数映射
    dist_funcs = {
        'euclidean': distance.euclidean,
        'cosine': distance.cosine,
        'correlation': distance.correlation,
        'canberra': distance.canberra,
        'minkowski': distance.minkowski,
        'chebyshev': distance.chebyshev,
        'braycurtis': distance.braycurtis,
        'cityblock': distance.cityblock,
    }
    
    # 转换动作向量
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy()
    else:
        action_np = action
    
    # 计算距离
    scores = []
    watched_movies = set(item['movie_id'] for item in user_history)
    
    for movie_id, emb in embeddings_dict.items():
        if movie_id == 0 or movie_id == "0":
            continue
        if movie_id in watched_movies:
            continue
        
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy()
        else:
            emb_np = emb
        
        dist = dist_funcs[metric](emb_np, action_np)
        scores.append([movie_id, dist])
    
    # 排序
    scores = sorted(scores, key=lambda x: x[1])
    top_k = scores[:k]
    
    # 计算相关性（这里使用简单的规则：如果用户历史中有相同类型的电影，则认为相关）
    # 注意：实际应用中应该使用更复杂的相关性计算
    relevance = []
    for movie_id, _ in top_k:
        # 简单相关性计算：如果电影ID在用户历史中，认为相关
        # 实际应用中应该基于电影类型、用户评分等计算相关性
        is_relevant = 1 if any(item['movie_id'] == movie_id for item in user_history) else 0
        relevance.append(is_relevant)
    
    # 计算评估指标
    ndcg = calculate_ndcg(relevance, k)
    precision = calculate_precision_at_k(relevance, k)
    
    return {
        'ndcg@k': ndcg,
        'precision@k': precision,
        'relevance': relevance
    }
