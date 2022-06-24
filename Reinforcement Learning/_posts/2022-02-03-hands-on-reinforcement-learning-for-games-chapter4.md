---
title: "Hands-On Reinforcement Learning for Games - Chapter 4"
excerpt: "Chapter 4: Temporal Difference Learning"
categories:
    - Reinforcement Learning
tags:
    - [RL, AI]
date: 2022-01-18
last_modified_at: 2022-01-18
---

# 1. 개요

**Hands-On Reinforcement Learning for Games** 강화 학습 책에 대해 공부한 내용을 정리하기 위한 포스트이기 때문에 자세히 기술하지 않을 것이다.      



# 2. TCA problem

**Temporal Credit Assignment (TCA)** 문제는 MC나 DP와는 다르게 time step에 걸쳐 최적 정책을 찾아야함. 즉, real-time으로 업데이트가 되야함.



# 3. Temporal Difference Learning

**Temporal Difference Learning (TDL)**은 *model-free* 이며 에피소드가 완전히 끝나지 않아도 됨. 즉, 알려지지 않은 환경을 real-time으로 탐험할 수 있음.

#### DP, MC, TDL에 대한 차이

* DP: 환경에 대한 정보를 미리 알고 있어야함
* MC: 수익 G는 에피소드가 끝나야만 계산 가능
* TD: 위 두 방법의 장점을 합침

![](/assets/images/backup-diagram-tdl.png)

## TD prediction

> $V(S_t) = V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

* $V(S_t)$: 현재 상태 가치
* $\alpha$: 학습률
* $R_{t+1}$: 다음 상태에 대한 보상
* $\gamma$: discount factor
* $V(S_{t+1})$: 다음 상태 가치



# 4. TDL을 Q-learning에 적용

**Q-learning**은 state-action pair를 학습함.  
시행착오를 통해 탐험하므로 model-free 속성을 가짐.  
다음 상태 중에서 행동가치가 가장 큰 행동의 행동가치 $\underset{a}{max}Q(S_{t+1}, a)$를 ***target*** 으로 함.

> $Q(S_t, A_t) = Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \underset{a}{max}Q(S_{t+1}, a) - Q(S_t, A_t)]$

* $Q(S_t, A_t)$: 현재 state-action quality

나머지 요소는 기존과 동일



# 5. TD(0) in Q-learning

Q-learning은 TD(0)의 단순화된 version임.

exploration과 exploitation 사이의 딜레마를 해결하기 위해 epsilon-greedy 방법에서의 $\epsilon$ 수치를 시간이 지날 수록 감소시킴.
환경이 크고 복잡할 수록 더 많이 exploration을 취해야할 필요가 있음.

* Random: 말 그래도 랜덤하게 선택. 새로운 환경에서의 효과적인 기준 테스트가 될 수 있음.
* Greedy: 항상 최상의 행동만을 선택하기 때문에 나쁜 결과를 초래할 수 있음.
* E-greedy: exploration을 균형적으로 수행할 수 있음.
* Bayesian or Thompson sampling: 샘플링된 작업의 무작위 분포를 통해 행동을 잘 선택하는 확률 및 통계를 사용함. 행동에 대한 모든 보상을 저장할 필요 없이 보상을 묘사하는 분포만 결정하면 됨.



# 6. Off-policy vs On-policy

TD(0), Q-learning은 policy or Q-table을 에피소드가 끝난 뒤 학습하기 때문에 off-policy 방법임.  
on-policy 방법으로는 *SARSA* 가 있음.