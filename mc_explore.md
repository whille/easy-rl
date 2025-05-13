# 基于探索性开始的蒙特卡洛方法，用于估算$\pi \approx \pi^*$

## 初始化
- $\pi(s) \in A(s)$ （随机初始化），对于所有的 $s \in S$  
- $Q(s,a) \in \mathbb{R}$ （随机初始化），对于所有的 $s \in S$，$a \in A(s)$  
- $R(s,a) \in \emptyset$ （初始化为空），对于所有的 $s \in S$，$a \in A(s)$  

## 回合循环
对每一个回合进行以下操作：  
1. 随机选择初始状态和动作：$s_0 \in S$，$a_0 \in A(s_0)$，且保证所有状态-动作对的概率 $>0$  
2. 从 $(s_0, a_0)$ 生成一个回合：$s_0, a_0, r_1, s_1, a_1, ..., s_{T-1}, a_{T-1}, r_T, s_T$  
3. 初始化累积奖励：$G \leftarrow 0$  

## 回合内循环（反向更新）
对于每一步 $t = T-1, T-2, \cdots, 0$：  
1. 更新累积奖励：$G \leftarrow \gamma G + r_{t+1}$  
2. 如果 $(s_t, a_t)$ 未出现在 $(s_0, a_0), (s_1, a_1), ..., (s_{t-1}, a_{t-1})$ 中：  
   - 将 $G$ 追加到 $R(s_t, a_t)$  
   - 更新动作价值函数：$Q(s_t, a_t) \leftarrow \text{average}(R(s_t, a_t))$  
   - 更新策略：$\pi(s_t) \leftarrow \arg\max_a Q(s_t, a)$  
