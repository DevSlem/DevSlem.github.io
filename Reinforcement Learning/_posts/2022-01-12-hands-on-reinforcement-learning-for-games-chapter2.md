---
title: "Hands-On Reinforcement Learning for Games - Chapter 2"
excerpt: "Chapter 2: Dynamic Programming and the Bellman Equation"
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




# 2. Dynamic Programming

**Dynamic Programming (DP)** 복잡한 결정 문제를 최적화하고 해결하기 위한 방법.  
먼저 하위 문제를 해결 후 이들을 연결하는 관계를 찾아 더 큰 문제를 해결하는 방법.  

> The term ***dynamic programming (DP)*** refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP)

DP는 Environment의 model(reward, state transition 등)에 대해 안다는 전제로 문제를 푸는 방법임.  
두개의 step으로 나뉨.

1. Prediction
2. Control

위 step을 처리하는 방식에 따라 [**Policy iteration**](#building-policy-iteration)과 [**Value iteration**](#building-value-iteration)으로 구분됨.

#### Memoization

DP의 핵심 컨셉은 앞서 말했듯이 큰 문제를 작은 하위 문제로 쪼개는 거임.  
쪼개진 **하위 문제들을 풀고 그 값들을 저장**하는데 이를 ***memoization***이라고 함.

#### 피보나치 수열

***memoization*** 활용의 대표적인 예시가 피보나치 수열 재귀호출 문제임.

> $\begin{aligned} F(n) = \begin{cases} \quad 0 & if \; n = 0 \\\ \quad 1 & if \; n = 1 \\\ F(n - 1) + F(n - 2) & if \; n > 1 \end{cases} \end{aligned}$

![피보나치 다이어그램](/assets/images/fibonacci-diagram.jpeg)

위 그림을 보면 알 수 있듯이 $n$이 커질 수록 중복 호출 및 계산되는 양이 늘어남.  
이를 해결하기 위해 이전에 계산했던 피보나치 수열의 값들을 저장하고 있을 필요가 있음.




# 3. Bellman equation

#### 벨만 방정식에서의 총 보상

벨만 방정식에서의 총 보상은 아래와 같음

> $R_{total} = R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots = \displaystyle\sum_{k = 0}^{\infty}{\gamma^k R_{t + k + 1}}$  
> 단, $0 \leq \gamma \leq 1$

$\gamma$는 ***discount factor(감가율)***라고 하며 총 보상을 구할 때 단순히 보상들의 합을 구하는 것이 아닌 ***discount factor***를 고려해야함. 
예컨데 우리가 어떤 행동을 취함으로써 얻는 경험이나 효과는 점점 미래로 멀어질 수록 약해지기 때문임.

#### 최적 정책

각 상태에 대한 값을 최대화하고, 어느 상태들을 거쳐야 보상을 최대화할 수 있는지를 결정하는 정책을 의미함.  
아래는 최적 정책을 구하는 수식임.  
참고로 $\mathbb{E}$는 기대값 기호임.

> $\pi(s) = \underset{a}{argmax}\mathbb{E}[R_{t + 1} + \gamma v_\pi(S_{t + 1}) \ \| \ S_t = s, \ A_t = a]$  
> $\pi(s) = \underset{a}{argmax} \displaystyle \sum_{s'}P(s'\|s,a)[R(s, a, s') + \gamma v_\pi(s')]$




# 4. Policy iteration

Policy Iteration은 **Policy evaluation(정책 평가)**와 **Policy improvement(정책 개선)**을 매번 반복하면서 최적의 정책을 찾는 방법임.  
Policy iteration에서 Policy evaluation은 **DP**의 **Prediction**에, Policy improvement는 **Control**에 해당됨. 

![정책반복](/assets/images/policy-iteration.png)

## Policy evaluation

#### 상태 가치 함수

> $v_{k+1} = \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1}) \ \| \ S_t = s]$  
> $v_{k+1} = \displaystyle\sum_{a}\pi(a\|s)\sum_{s'}P(s'\|s,a)[R(s,a,s') + \gamma v_\pi(s')]$

#### 위 식의 요소

* $v$: 상태 가치
* $R$: 보상
* $s$: 상태
* $\pi$: 정책
* $P$: 상태 전이 확률(확률론적 환경)
* $\gamma$: discount factor


#### Backup Diagram

> **Backup**: 미래의 값들(next state-value function)로 현재의 value function을 구하는 것

위 [**상태 가치 함수 식**](#상태-가치-함수)을 보면 어떤 action을 취했을 때 얻게되는 보상과, 전이된 next state에서의 $v$(상태 가치)를 통해 현재의 상태 가치를 구하고 있음. 이것이 **Backup** 과정임.  
이 때, 전이된 next state에서의 상태 가치는, 이전 iteration에서 구한 후 저장해 놓고 있던 상태 가치 데이터로 앞에서 언급한 ***memoization***  기법임.  
아래 그림은 **Backup** 과정을 보여줌.

![정책반복 백업](/assets/images/policy-iteration-backup-diagram.png)

## Policy improvement

#### 행동 가치 함수

> $q(s,a) = \displaystyle\sum_{s'}P(s'\|s,a)[R(s,a,s') + \gamma v_\pi(s')]$

#### 정책 개선 함수

정책 개선은 행동 가치가 최대가 되는 **action(행동)**을 선택하는 것.

> $\begin{aligned} \pi(s) &= \underset{a}{argmax} \displaystyle \sum_{s'}P(s'\|s,a)[R(s, a, s') + \gamma v_\pi(s')] \\\ &= \underset{a}{argmax}\Bigl(q(s,a)\Bigl) \end{aligned}$




# 5. Value iteration

Value iteration은 일단 **Optimal state value(최적 상태 가치)**을 구한 후 정책을 추출하는 방법임.  
Value iteration에서는 **Optimal state value**를 찾는 과정이 **DP**의 **Prediction**에, **Policy extraction(정책 추출)**이 **Control**에 해당됨.  
Policy iteration의 경우 정책 평가와 정책 개선을 번갈아 반복하기 때문에 상태 가치와 정책이 모두 수렴할 때까지 반복하지만, Value iteration은 일단 상태 가치를 수렴시킨 후 정책을 추출한다는 차이가 있음.

![값반복](/assets/images/value-iteration.png)

#### 상태 가치 함수

> $\begin{aligned} v_{k+1} &= \underset{a}{\max}\sum_{s'}P(s'\|s,a)[R(s,a,s') + \gamma v_k(s')] \\\ &= \underset{a}{\max}\Bigl(q(s,a)\Bigl) \end{aligned}$

위 식을 보면 상태 가치를 구하는데 있어 Policy iteration과 차이가 있음.  
Policy iteration에서는 행동 가치들의 **기대값**을 통해 구하지만, Value iteration에서는 행동 가치 중 **최대값**을 구함.

#### 정책 추출

정책을 구하는 방법은 위에서 봤던 것과 동일함. 주의할 점은 Policy iteration은 매 iteration마다 정책을 구하지만, Value iteration에서는 최적 상태 가치가 수렴했을 때에만 정책을 구하며 이는 곧 최적 정책임.

