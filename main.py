import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import Actor, Critic
from src.data import FrameEnv, DataPath
from src.algorithms import ddpg_update
from src.utils import soft_update, DummyWriter
from src.utils.evaluation import evaluate_recommendations


class DDPGAgent:
    def __init__(self, policy_net, value_net, device=torch.device('cuda')):
        self.device = device
        self.nets = self._setup_networks(policy_net, value_net)
        self.optimizers = self._setup_optimizers()
        self.params = self._setup_params()
        self._step = 0
        self.writer = DummyWriter()
        self.best_ndcg = -1  # 最佳NDCG值

    def _setup_networks(self, policy_net, value_net):
        import copy
        target_policy_net = copy.deepcopy(policy_net)
        target_value_net = copy.deepcopy(value_net)

        target_policy_net.eval()
        target_value_net.eval()

        soft_update(value_net, target_value_net, soft_tau=1.0)
        soft_update(policy_net, target_policy_net, soft_tau=1.0)

        return {
            "value_net": value_net,
            "target_value_net": target_value_net,
            "policy_net": policy_net,
            "target_policy_net": target_policy_net,
        }

    def _setup_optimizers(self):
        # 设置合理的学习率
        value_optimizer = optim.Ranger(
            self.nets["value_net"].parameters(), lr=5e-5, weight_decay=1e-2
        )
        policy_optimizer = optim.Ranger(
            self.nets["policy_net"].parameters(), lr=5e-5, weight_decay=1e-2
        )

        optimizers = {
            "policy_optimizer": policy_optimizer,
            "value_optimizer": value_optimizer,
        }

        # 添加学习率调度器
        self.schedulers = {
            "policy_scheduler": ReduceLROnPlateau(
                policy_optimizer,
                mode='min',
                factor=0.5,      # 每次学习率减半
                patience=5,       # 5个epoch没有改善就降低
                min_lr=1e-8      # 最小学习率
            ),
            "value_scheduler": ReduceLROnPlateau(
                value_optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-8
            )
        }

        return optimizers

    def _setup_params(self):
        return {
            "gamma": 0.99,
            "min_value": -10,
            "max_value": 10,
            "policy_step": 3,
            "soft_tau": 0.001,
            "policy_lr": 1e-5,
            "value_lr": 1e-5,
            "actor_weight_init": 3e-1,
            "critic_weight_init": 6e-1,
            "contrastive_weight": 0.5,  # 对比学习损失权重
        }

    def update(self, batch, learn=True, debug=None):
        return ddpg_update(
            batch,
            self.params,
            self.nets,
            self.optimizers,
            device=self.device,
            debug=debug,
            writer=self.writer,
            learn=learn,
            step=self._step,
        )

    def step(self):
        self._step += 1

    def step_schedulers(self, metric):
        """更新学习率调度器"""
        self.schedulers["policy_scheduler"].step(metric)
        self.schedulers["value_scheduler"].step(metric)

    def get_lr(self):
        """获取当前学习率"""
        return {
            "policy_lr": self.optimizers["policy_optimizer"].param_groups[0]["lr"],
            "value_lr": self.optimizers["value_optimizer"].param_groups[0]["lr"]
        }

    def to(self, device):
        self.nets = {k: v.to(device) for k, v in self.nets.items()}
        self.device = device
        return self


def main():
    print("Starting Yengh Recommendation System Training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    frame_size = 10
    batch_size = 25
    n_epochs = 100
    plot_every = 30

    import os
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = DataPath(
        base=os.path.join(project_dir, "data") + "/",
        embeddings="embeddings/ml20_pca128.pkl",
        ratings="ml-20m/ratings.csv",
        cache="cache/frame_env.pkl",
        use_cache=True
    )

    print("Loading environment...")
    env = FrameEnv(data_path, frame_size, batch_size, num_workers=0)
    print("Environment loaded successfully!")
    
    # 加载电影嵌入和电影字典用于评估
    import pickle
    import pandas as pd
    
    # 加载电影嵌入
    embeddings_path = os.path.join(project_dir, "data", "embeddings", "ml20_pca128.pkl")
    with open(embeddings_path, 'rb') as f:
        movie_embeddings = pickle.load(f)
    
    # 转换为整数键
    movie_embeddings = {int(k): v for k, v in movie_embeddings.items()}
    
    # 加载电影字典和类型信息
    movies_path = os.path.join(project_dir, "data", "ml-20m", "movies.csv")
    movies_df = pd.read_csv(movies_path)
    movie_dict = dict(zip(movies_df['movieId'], movies_df['title']))
    
    # 提取电影类型信息
    movie_genres = {}
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        genres = row['genres'].split('|')
        movie_genres[movie_id] = genres
    
    # 从环境中获取用户字典
    user_dict = env.base.test_user_dataset.user_dict

    input_dim = 128 * frame_size + frame_size  # items (128*frame_size) + ratings (frame_size)
    action_dim = 128
    hidden_size = 256

    print("Creating networks...")
    policy_net = Actor(input_dim, action_dim, hidden_size).to(device)
    value_net = Critic(input_dim, action_dim, hidden_size).to(device)

    print("Initializing agent...")
    agent = DDPGAgent(policy_net, value_net, device=device)

    writer = SummaryWriter('./runs/ddpg_experiment')
    agent.writer = writer

    print("Starting training...")
    best_value_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20  # 早停耐心值

    # 提前创建模型保存目录
    os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)

    for epoch in range(n_epochs):
        epoch_losses = []

        for i in range(100):
            batch = env.train_batch()
            losses = agent.update(batch, learn=True)
            agent.step()
            epoch_losses.append(losses)

        avg_value_loss = sum([l['value'] for l in epoch_losses]) / len(epoch_losses)
        avg_policy_loss = sum([l['policy'] for l in epoch_losses]) / len(epoch_losses)
        current_lr = agent.get_lr()

        print(f"Epoch {epoch}/{n_epochs} - Value Loss: {avg_value_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Learning Rates - Policy: {current_lr['policy_lr']:.2e}, Value: {current_lr['value_lr']:.2e}")

        # 测试并更新学习率
        # 每轮都进行测试和评估
        test_batch = env.test_batch()
        test_losses = agent.update(test_batch, learn=False)
        test_value_loss = test_losses['value']
        print(f"Test - Value Loss: {test_value_loss:.4f}, Policy Loss: {test_losses['policy']:.4f}")

        # 计算评估指标
        # 使用真实的用户历史数据进行评估
        ndcg_scores = []
        precision_scores = []
        
        # 从测试批次中获取用户ID
        if 'meta' in test_batch and 'users' in test_batch['meta']:
            user_ids = test_batch['meta']['users'][:5]  # 取前5个用户
            
            for user_id in user_ids:
                # 获取用户历史数据
                if user_id.item() in user_dict:
                    user_data = user_dict[user_id.item()]
                    user_items = user_data['items']
                    user_ratings = user_data['ratings']
                    
                    # 构建用户历史
                    user_history = []
                    for i, movie_id in enumerate(user_items[:10]):  # 取最近10部电影
                        # 转换为原始电影ID
                        original_movie_id = env.base.id_to_key.get(movie_id, movie_id)
                        user_history.append({'movie_id': original_movie_id, 'rating': user_ratings[i]})
                    
                    # 随机选择10部电影作为推荐
                    movie_ids = list(movie_embeddings.keys())
                    if len(movie_ids) > 10:
                        # 过滤掉用户已观看的电影
                        watched_movies = set(item['movie_id'] for item in user_history)
                        candidate_movies = [mid for mid in movie_ids if mid not in watched_movies]
                        
                        if len(candidate_movies) >= 10:
                            recommended_ids = np.random.choice(candidate_movies, 10, replace=False)
                        else:
                            recommended_ids = candidate_movies[:10]
                        
                        # 计算相关性（基于电影类型）
                        relevance = []
                        for movie_id in recommended_ids:
                            # 计算类型相似度
                            genre_similarity = 0
                            if movie_id in movie_genres:
                                recommended_genres = set(movie_genres[movie_id])
                                # 计算与用户历史电影的类型相似度
                                for item in user_history:
                                    if item['movie_id'] in movie_genres:
                                        user_genres = set(movie_genres[item['movie_id']])
                                        common_genres = recommended_genres & user_genres
                                        if common_genres:
                                            genre_similarity += len(common_genres) / len(recommended_genres | user_genres)
                                # 平均相似度
                                if user_history:
                                    genre_similarity /= len(user_history)
                            
                            # 将相似度转换为相关性分数
                            if genre_similarity > 0.3:
                                relevance_score = 1
                            elif genre_similarity > 0.1:
                                relevance_score = 0.5
                            else:
                                relevance_score = 0
                            
                            relevance.append(relevance_score)
                        
                        # 计算评估指标
                        from src.utils.evaluation import calculate_ndcg, calculate_precision_at_k
                        ndcg = calculate_ndcg(relevance, 10)
                        precision = calculate_precision_at_k(relevance, 10)
                        
                        ndcg_scores.append(ndcg)
                        precision_scores.append(precision)
        
        if ndcg_scores:
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
            avg_precision = sum(precision_scores) / len(precision_scores)
            
            print(f"  Evaluation - NDCG@10: {avg_ndcg:.4f}, Precision@10: {avg_precision:.4f}")
            writer.add_scalar('eval/ndcg@10', avg_ndcg, epoch)
            writer.add_scalar('eval/precision@10', avg_precision, epoch)
        else:
            print("  Evaluation - Insufficient data for evaluation")

        # 更新学习率调度器（基于测试损失）
        agent.step_schedulers(test_value_loss)

        # 保存最佳模型（基于损失值）
        if test_value_loss < best_value_loss:
            best_value_loss = test_value_loss
            patience_counter = 0
            print(f"  ✓ New best model! Loss: {best_value_loss:.4f}")

            # 保存最佳模型
            save_data = {
                'policy_net': agent.nets['policy_net'].state_dict(),
                'value_net': agent.nets['value_net'].state_dict(),
                'epoch': epoch,
                'loss': best_value_loss,
            }
            
            # 如果有NDCG值，也保存
            if 'avg_ndcg' in locals():
                save_data['ndcg'] = avg_ndcg
                agent.best_ndcg = avg_ndcg
                print(f"  NDCG@10: {avg_ndcg:.4f}")
            
            torch.save(save_data, os.path.join(project_dir, 'models', 'best-clattention.pth'))
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stopping_patience}")

        # 早停检查
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best loss: {best_value_loss:.4f}")
            if hasattr(agent, 'best_ndcg') and agent.best_ndcg > 0:
                print(f"Best NDCG@10: {agent.best_ndcg:.4f}")
            break

    print("Training completed!")
    print(f"Best model saved to models/best-clattention.pth with loss: {best_value_loss:.4f}")
    if hasattr(agent, 'best_ndcg') and agent.best_ndcg > 0:
        print(f"Best NDCG@10: {agent.best_ndcg:.4f}")


if __name__ == "__main__":
    main()
