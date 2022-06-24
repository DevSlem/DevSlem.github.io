---
title: "Hands-On Reinforcement Learning for Games - Chapter 1"
excerpt: "Chapter 1: Understanding Rewards-Based Learning"
categories:
    - Reinforcement Learning
tags:
    - [RL, AI]
date: 2022-01-12
last_modified_at: 2022-01-12
---


# 1. 개요

**Hands-On Reinforcement Learning for Games** 강화 학습 책에 대해 공부한 내용을 정리하기 위한 포스트이기 때문에 자세히 기술하지 않을 것이다.  
참고로 이 책은 개념 위주가 아닌 실습 위주의 책이기 떄문에 개념을 자세히 설명해주지 않는다.

![책](https://static.packt-cdn.com/products/9781839214936/cover/smaller)

> 책 링크: [**Buy from Packt**](https://www.packtpub.com/product/hands-on-reinforcement-learning-for-games/9781839214936)  
> 코드 링크: [**Code from Github**](https://github.com/packtpublishing/hands-on-reinforcement-learning-for-games)




# 2. Understanding rewards-based learning

#### Variations of supervised learning

| 종류                     | Training Data                          |
| ------------------------ | -------------------------------------- |
| Supervised Learning      | All Labelled Data                      |
| Semi-Supervised Learning | Some Labelled Data  Unlabeled Data |
| Unsupervised Learning    | Unlabeled Data                         |


**Reinforcement Learning(RL)** 문제를 풀 때 semi-supervised learning을 주로 사용한다고 함.

#### 기본 용어

* Agent: RL system;
* Environment: 게임 보드, 게임 스크린 등
* State: 환경의 최근 상태의 스냅샷
* Reward: 환경에 의해 제공되는 것으로, Agent에게 좋거나 나쁜 피드백을 줌
* Action: Agent가 취할 수 있는 행동

#### RL의 요소

* Policy: 특정 state에서 agent가 취할 행동에 대한 planning process
* Reward function: 보상의 양을 결정
* Value function: 장기간에 걸친 상태의 가치
* Model: 환경 그자체(게임에서는 가능한 모든 게임 상태)




# 3. The Markov decision process

#### The Markov property and MDP

**Markov Property**: 미래 상태의 조건부 확률 분포가 현재 상태에 의해서만 결정됨  
Markov signal 혹은 state는 agent가 특정 상태로부터 value를 예측하는 경우 Markov property로 여겨짐.  
모든 Markov signal 혹은 state가 미래 상태를 예측할 수 있으면 RL 문제는 Markov property를 충족함.

Markov property이고 유한한 learning task를 finite **Markov decision process(MDP)**라고 함.




# 4. Value learning with multi-armed bandits

**multi-armed bandits**: agent가 슬롯 머신의 팔을 내렸을 때 보상을 최대화하기 위해 어떤 슬롯을 내려야 하는지를 추정하는 책에서 정의한 문제. 이 포스트에서는 기술하지 않겠음.

#### Value equation

> $V(a) = V(a) + \alpha(r - V(a))$  
> $V(a) = (1 - \alpha)V(a) + \alpha r$

#### 위 식의 요소

* $V(a)$: the value for a given action(old value)
* $a$: action
* $\alpha$: the learning rate
* $r$: reward(new/learned value)

#### $\alpha$에 대한 해석

위 식중 2번째 식을 보면 $\alpha$에 대해 아래와 같이 해석할 수 있음.

> $\alpha$가 커질 수록 즉각적 보상에 더 가중치를 부여함

#### Greedy Policy

가장 행동가치가 높은 행동만 선택

문제점: 순간적으로 좋아보이는 행동만 선택하기 때문에 더 먼 미래를 고려하지 못함. 즉, 학습 능력을 심각하게 제한함. 다양한 경험을 하지 못한다고 생각하면 됨.




# 5. Q-learning with contextual bandits

#### Q-learning equation

> $Q(s, a) = Q(s, a) + \alpha[r + \gamma \underset{a'}{\max}(s', a') - Q(s, a)]$

#### 위 식의 요소

* $s$: state
* $s'$: next state
* $a$: action
* $a'$: next action
* $\gamma$: discount factor
* $\alpha$: learning rate
* $r$: reward