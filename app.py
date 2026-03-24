import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import random
import torch
from scipy.spatial import distance
import sys
import os

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.models import Actor, Critic, AttentionMechanism
from src.data import FrameEnv, DataPath
from src.utils import soft_update

st.set_page_config(
    page_title="Yengh Recommendation System",
    page_icon="🎬",
    layout="wide"
)


def render_header():
    st.title("🎬 Yengh Recommendation System")
    st.markdown("""
    ### 基于强化学习的个性化推荐系统
    
    这是一个基于DDPG算法的电影推荐系统演示。
    
    **功能特点：**
    - 🤖 使用深度强化学习生成推荐
    - 📊 支持多种距离度量方式
    - 🎯 可视化推荐结果
    - ⚡ 实时交互体验
    """)


@st.cache_resource
def load_model_and_data():
    """加载模型和数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model_path = os.path.join(BASE_DIR, "models", "best-clattention.pth")
    policy_net = Actor(1290, 128, 256).to(device)
    
    if os.path.exists(model_path):
        # 在PyTorch 2.6中，weights_only默认值为True，需要设置为False来加载包含numpy标量的模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net'])
        else:
            policy_net.load_state_dict(checkpoint)
        policy_net.eval()
    
    # 初始化注意力模型
    attention_model = AttentionMechanism(128, 64).to(device)
    
    # 加载电影数据
    movies_path = os.path.join(BASE_DIR, "data", "ml-20m", "movies.csv")
    movies_df = pd.read_csv(movies_path)
    movie_dict = dict(zip(movies_df['movieId'], movies_df['title']))
    
    # 加载评分数据
    ratings_path = os.path.join(BASE_DIR, "data", "ml-20m", "ratings.csv")
    ratings_df = pd.read_csv(ratings_path)
    
    # 加载电影嵌入
    embeddings_path = os.path.join(BASE_DIR, "data", "embeddings", "ml20_pca128.pkl")
    with open(embeddings_path, 'rb') as f:
        movie_embeddings = pickle.load(f)
    
    # 创建电影ID到索引的映射
    movie_ids = sorted(movie_embeddings.keys())
    id_to_idx = {int(movie_id): idx for idx, movie_id in enumerate(movie_ids)}
    
    embeddings_tensor = torch.stack([movie_embeddings[mid] for mid in movie_ids])
    
    movie_embeddings_int_key = {int(k): v for k, v in movie_embeddings.items()}
    
    return policy_net, attention_model, device, movie_dict, ratings_df, embeddings_tensor, id_to_idx, movie_ids, movie_embeddings_int_key


def get_user_history(user_id, ratings_df, movie_dict, n=10):
    """获取用户的历史观影记录（按时间正序，与训练一致）"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('timestamp', ascending=True)
    if len(user_ratings) == 0:
        return None
    
    recent = user_ratings.tail(n)
    history = []
    for _, row in recent.iterrows():
        movie_id = int(row['movieId'])
        rating = row['rating']
        title = movie_dict.get(movie_id, f"Movie {movie_id}")
        history.append({
            'movie_id': movie_id,
            'title': title,
            'rating': rating
        })
    
    return history


def build_state_from_history(history, embeddings_tensor, id_to_idx, frame_size=10, attention_model=None):
    """从历史记录构建状态向量
    
    状态向量结构：
    - 前1280维：frame_size个物品的嵌入向量（每个128维）
    - 后10维：frame_size个评分值（归一化到[-2.5, 2.5]）
    """
    embedding_dim = 128
    state_dim = embedding_dim * frame_size + frame_size  # 1280 + 10 = 1290
    state = torch.zeros(state_dim, dtype=torch.float32)
    
    if not history or len(history) == 0:
        return state.unsqueeze(0)
    
    valid_count = 0
    for i, item in enumerate(history[:frame_size]):
        movie_id = item['movie_id']
        rating = item['rating']
        
        if movie_id in id_to_idx:
            idx = id_to_idx[movie_id]
            if idx < embeddings_tensor.size(0):
                state[i * embedding_dim : (i + 1) * embedding_dim] = embeddings_tensor[idx]
                state[embedding_dim * frame_size + i] = 2 * (rating - 2.5)
                valid_count += 1
        else:
            print(f"Warning: movie_id {movie_id} not found in id_to_idx")
    
    print(f"State built: valid_count={valid_count}/{min(len(history), frame_size)}, state_norm={state.norm().item():.4f}")
    
    return state.unsqueeze(0)


def recommend_movies(action, embeddings_dict, movie_dict, k=10, metric='euclidean', watched_movies=None):
    """根据动作向量推荐电影
    
    参考原库实现：使用scipy.spatial.distance计算距离
    
    :param action: 动作向量（128维）
    :param embeddings_dict: 电影嵌入字典 {movie_id: embedding}
    :param movie_dict: 电影名称字典 {movie_id: title}
    :param k: 推荐数量
    :param metric: 距离度量方式
    :param watched_movies: 已观看的电影ID集合，用于过滤
    """
    from scipy.spatial import distance
    from scipy.spatial.distance import euclidean, cosine, correlation, canberra, minkowski, chebyshev, braycurtis, cityblock
    
    if watched_movies is None:
        watched_movies = set()
    
    dist_funcs = {
        'euclidean': euclidean,
        'cosine': cosine,
        'correlation': correlation,
        'canberra': canberra,
        'minkowski': minkowski,
        'chebyshev': chebyshev,
        'braycurtis': braycurtis,
        'cityblock': cityblock,
    }
    
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy()
    else:
        action_np = action
    
    scores = []
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
    
    # 排序并取前k个
    scores = sorted(scores, key=lambda x: x[1])
    scores = scores[:k]
    
    recommendations = []
    for movie_id, dist in scores:
        title = movie_dict.get(movie_id, f"Movie {movie_id}")
        recommendations.append({
            'movie_id': movie_id,
            'title': title,
            'distance': dist,
            'similarity': 1 / (1 + dist)  # 转换为相似度
        })
    
    return recommendations


def generate_recommendations(state, model, k=10):
    """生成推荐"""
    with torch.no_grad():
        action = model(state)
    return action


def calculate_distances(action, embeddings, metric='euclidean'):
    """计算距离"""
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
    
    scores = []
    for i, emb in enumerate(embeddings):
        scores.append([i, dist_funcs[metric](emb, action)])
    
    scores = sorted(scores, key=lambda x: x[1])
    return scores


def main():
    render_header()
    
    # 侧边栏配置
    st.sidebar.header("⚙️ 配置")
    
    # 设备选择
    use_cuda = st.sidebar.checkbox("使用GPU", torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    st.sidebar.write(f"当前设备: {device}")
    
    # 页面选择
    page = st.sidebar.selectbox(
        "选择功能",
        ["🏠 首页", "🤖 模型测试", "📊 推荐演示"]
    )
    
    if page == "🏠 首页":
        st.markdown("""
        ## 欢迎使用 RecNN 推荐系统
        
        ### 快速开始
        
        1. **准备数据**: 下载MovieLens 20M数据集和电影嵌入文件
        2. **训练模型**: 运行 `python main.py` 训练DDPG模型
        3. **启动应用**: 运行 `streamlit run app.py`
        
        ### 功能说明
        
        - **模型测试**: 测试训练好的推荐模型
        - **推荐演示**: 交互式推荐演示
        
        ### 数据要求
        
        需要以下数据文件：
        - `models/best_model.pth` - 训练好的最佳模型
        """)
    
    elif page == "🤖 模型测试":
        st.header("模型测试")
        
        # 加载模型
        try:
            policy_net, attention_model, device, movie_dict, ratings_df, embeddings_tensor, id_to_idx, movie_ids, movie_embeddings = load_model_and_data()
            st.success("✅ 模型加载成功！")
        except Exception as e:
            st.error(f"加载失败: {e}")
            return
        
        # 生成随机状态
        st.subheader("生成推荐")
        
        batch_size = st.slider("批次大小", 1, 50, 10)
        
        # 生成随机状态
        # state = items (128*10) + ratings (10) = 1290
        input_dim = 128 * 10 + 10  # frame_size * embedding_dim + frame_size
        state = torch.randn(batch_size, input_dim).to(device)
        
        # 生成动作
        with torch.no_grad():
            action = policy_net(state)
        
        st.write(f"状态形状: {state.shape}")
        st.write(f"动作形状: {action.shape}")
        
        # 显示动作统计
        st.subheader("动作统计")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("动作均值", f"{action.mean().item():.4f}")
        with col2:
            st.metric("动作标准差", f"{action.std().item():.4f}")
        with col3:
            st.metric("动作范围", f"[{action.min().item():.2f}, {action.max().item():.2f}]")
        
        # 可视化
        st.subheader("动作分布")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 直方图
        axes[0].hist(action.cpu().numpy().flatten(), bins=50, alpha=0.7)
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Action Distribution")
        
        # 热力图
        im = axes[1].imshow(action.cpu().numpy()[:20], aspect='auto', cmap='viridis')
        axes[1].set_xlabel("Embedding Dimension")
        axes[1].set_ylabel("Batch Index")
        axes[1].set_title("Action Heatmap (First 20)")
        plt.colorbar(im, ax=axes[1])
        
        st.pyplot(fig)
    
    elif page == "📊 推荐演示":
        st.header("🎯 智能推荐演示")
        
        st.info("基于DDPG强化学习算法的个性化电影推荐")
        
        # 加载模型和数据
        with st.spinner("正在加载模型和数据..."):
            try:
                policy_net, attention_model, device, movie_dict, ratings_df, embeddings_tensor, id_to_idx, movie_ids, movie_embeddings = load_model_and_data()
                st.success("✅ 模型和数据加载成功！")
            except Exception as e:
                st.error(f"加载失败: {e}")
                st.info("请确保已运行 `python main.py` 训练模型")
                return
        
        # 获取所有用户ID
        all_users = sorted(ratings_df['userId'].unique())
        
        # 用户选择
        st.subheader("👤 选择用户")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 随机选择或手动输入
            select_method = st.radio("选择方式", ["随机用户", "指定用户ID"])
            
            if select_method == "随机用户":
                if st.button("🎲 随机选择一个用户"):
                    selected_user = random.choice(all_users)
                    st.session_state['selected_user'] = selected_user
                selected_user = st.session_state.get('selected_user', all_users[0])
            else:
                selected_user = st.number_input(
                    "输入用户ID",
                    min_value=int(min(all_users)),
                    max_value=int(max(all_users)),
                    value=int(all_users[0])
                )
        
        with col2:
            st.metric("当前用户ID", selected_user)
            user_movie_count = len(ratings_df[ratings_df['userId'] == selected_user])
            st.metric("观影数量", user_movie_count)
        
        # 获取用户历史
        history = get_user_history(selected_user, ratings_df, movie_dict, n=10)
        
        if history is None:
            st.warning("该用户没有观影记录，请选择其他用户")
            return
        
        # 显示用户历史
        st.subheader("📺 用户历史观影记录（最近10部）")
        
        history_df = pd.DataFrame(history)
        history_df['评分'] = history_df['rating'].apply(lambda x: "⭐" * int(x))
        
        st.dataframe(
            history_df[['title', 'rating', '评分']].rename(columns={
                'title': '电影名称',
                'rating': '评分（1-5）'
            }),
            width='stretch'
        )
        
        # 推荐配置
        st.subheader("⚙️ 推荐配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("推荐数量", 1, 20, 10)
        
        with col2:
            st.write("推荐算法：DDPG强化学习")
            st.write("状态维度：1290（1280+10）")
            st.write("动作维度：128")
        
        # 生成推荐
        if st.button("🚀 生成个性化推荐", type="primary"):
            with st.spinner("🤖 AI正在分析用户偏好并生成推荐..."):
                state = build_state_from_history(history, embeddings_tensor, id_to_idx, attention_model=attention_model)
                state = state.to(device)
                
                with torch.no_grad():
                    action = policy_net(state).squeeze()
                
                print(f"Action stats: mean={action.mean().item():.4f}, std={action.std().item():.4f}, norm={action.norm().item():.4f}")
                
                watched_movies = set(item['movie_id'] for item in history)
                
                recommendations = recommend_movies(
                    action, movie_embeddings, movie_dict, k=top_k, watched_movies=watched_movies
                )
                
                # 显示推荐结果
                st.subheader("🎬 为您推荐的电影")
                
                rec_df = pd.DataFrame(recommendations)
                rec_df['排名'] = range(1, len(rec_df) + 1)
                rec_df['匹配度'] = rec_df['similarity'].apply(lambda x: f"{x*100:.1f}%")
                
                # 美化显示
                st.dataframe(
                    rec_df[['排名', 'title', '匹配度']].rename(columns={
                        'title': '电影名称'
                    }),
                    width='stretch',
                    hide_index=True
                )
                
                # 可视化
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 匹配度分布")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = plt.cm.RdYlGn(rec_df['similarity'])
                    bars = ax.barh(rec_df['title'], rec_df['similarity'], color=colors)
                    ax.set_xlabel('匹配度（相似度）')
                    ax.set_ylabel('电影名称')
                    ax.set_title('推荐电影匹配度')
                    ax.set_xlim(0, 1)
                    
                    # 添加数值标签
                    for i, (bar, sim) in enumerate(zip(bars, rec_df['similarity'])):
                        ax.text(sim + 0.01, i, f'{sim*100:.1f}%', 
                               va='center', fontsize=9)
                    
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("🎯 兴趣向量可视化")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    action_np = action.cpu().numpy()
                    ax.plot(action_np, linewidth=2, color='steelblue')
                    ax.fill_between(range(len(action_np)), action_np, alpha=0.3)
                    ax.set_xlabel('维度')
                    ax.set_ylabel('值')
                    ax.set_title('用户兴趣向量（128维）')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # 推荐解释
                st.subheader("💡 推荐解释")
                st.markdown(f"""
                **推荐逻辑：**
                - 基于您最近观看的 **{len(history)}** 部电影
                - AI模型分析了您的观影偏好，生成了一个 **128维** 的兴趣向量
                - 从 **{len(movie_ids)}** 部电影中筛选出与您兴趣最匹配的 **{top_k}** 部
                
                **匹配度说明：**
                - 匹配度越高，表示该电影与您的兴趣偏好越相似
                - 算法综合考虑了电影类型、风格、年代等多个维度
                """)


if __name__ == "__main__":
    main()
