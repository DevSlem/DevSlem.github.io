---
title: "Dynamic Programming"
excerpt: "MDP에서 optimal policy를 계산하는데 사용되는 DP를 소개한다."
tags: [RL, AI]
date: 2022-06-25
last_modified_at: 2022-06-25
sidebar:
    nav: "rl"
---

이 포스트에서는 MDP로써 environment의 perfect model이 주어졌을 때 optimal policy를 계산하는데 사용되는 기초적인 방식인 Dynamic Programming (DP)를 소개한다.

## Introduction

*Dynamic Programming* (DP)는 복잡한 문제를 간단한 여러 개의 문제로 나누어 푸는 최적화 기법이다. DP는 일반적으로 아래와 같은 2가지 유형의 문제에 적용된다.

1. [Overlapping Subproblems](https://www.geeksforgeeks.org/overlapping-subproblems-property-in-dynamic-programming-dp-1/)
2. [Optimal Substructure](https://www.geeksforgeeks.org/overlapping-subproblems-property-in-dynamic-programming-dp-1/)

Reinforcement Learning (RL)에서 environment가 perfect model로 environment의 지식 (일반적으로 MDP)을 완전히 알 수 있다면 DP를 활용하여 optimal policy를 계산할 수 있다. RL에서 DP의 핵심은 Bellman equation을 value function의 근사를 위한 update rule로 전환하는 것이다. 이 과정이 어떻게 진행되는지 알아보자.

## Generalized Policy Iteration

*Generalized Policy Iteration* (GPI)는 RL 문제를 풀 때 사용되는 일반적인 접근 방법이다. GPI에는 다음과 같은 2가지 과정이 존재한다.

1. Policy Evaluation - policy $\pi$를 따를 때 value function $v_\pi$를 계산
2. Policy Improvement - 계산된 현재 value function을 통해 policy $\pi$를 개선

위 policy evaluation과 policy improvement는 번갈아 가며 수행된다. 아래 다이어그램을 보면 조금 더 직관적으로 이해할 수 있다.

![](/assets/images/rl-sutton-gpi-diagram.png){: w="30%"}
_Fig 1. GPI diagram.  
(Image source: Sec. 4.6 Sutton & Barto (2017).)_

policy evaluation과 policy improvement는 서로 경쟁하면서 협력하는 관계로 볼 수 있다. 이 둘은 서로를 반대 방향으로 잡아 당긴다. policy evaluation에서는 현재 policy $\pi$에 관해 value function $v_\pi$가 계산된다. 즉, value function을 policy 쪽으로 끌어당긴 셈이다. 반대로 policy improvement에서는 value function $v_\pi$에 관해 policy $\pi$가 개선되기 때문에 policy를 value function 쪽으로 끌어 당겼다. 이렇게 서로 끌어당기다 보면 어느 시점에 한 지점에 도달하게 되고 이 때가 바로 optimal value function과 policy이다. 이러한 관계를 나타내는 그림은 아래와 같다.

![](/assets/images/rl-sutton-gpi-relationship.png){: w="50%"}
_Fig 2. GPI relationship.  
(Image source: Sec. 4.6 Sutton & Barto (2017).)_

몰론 실제로는 엄청나게 복잡한 과정이 내부에서 발생하지만 직관적으로 위와 같이 GPI를 이해할 수 있다. GPI에 대해 알아보았으니 이제 DP에서 GPI가 어떻게 적용되는지 알아보자.

## Policy Evaluation (Prediction)

DP에서의 *policy evaluation*에 대해 알아보자. policy evaluation은 *prediction*으로 불리기도 한다. 먼저 Bellman equation을 리마인드하자.

$$
\begin{align}
    v_\pi(s) &\doteq \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \ \vert \ S_t = s] \\
    &= \sum_a\pi(a \vert s) \sum_{s',r}p(s',r \vert s,a) \Big[r + \gamma v_\pi(s') \Big] \tag{1}
\end{align}
$$

DP에서는 위 Bellman equation을 아래와 같은 연속적인 update rule로 적용해 value function $v(s)$를 풀 수 있다.

$$
\begin{align}
    v_{k+1}(s) &\doteq \mathbb{E}_\pi[R_{t+1} + \gamma v_k(S_{t+1}) \ \vert \ S_t = s] \\
    &= \sum_a\pi(a \vert s) \sum_{s',r}p(s',r \vert s,a) \Big[r + \gamma v_\pi(s') \Big] \tag{2}
\end{align}
$$

DP에서는 위 update rule이 모든 state $s \in \mathcal{S}$에 대해 수행되며 $v_k$는 일반적으로 $k \rightarrow \infty$이면 수렴한다. 이러한 알고리즘을 *iterative policy evaluation*이라고 부른다.

iterative policy evaluation이 DP인 이유는 update rule을 수행하는 방식 때문이다. 기존 Bellman equation (1)에서는 현재 state $s$의 value $v_\pi(s)$는 transition된 next state $s'$의 $v_\pi(s')$으로부터 계산된다. $v_\pi(s')$은 다시 transition된 s'의 후속 state $s''$의 $v_\pi(s'')$로부터 계산된다. 즉, 재귀적 관계이다. 이러한 방식은 당연히 비효율적이며 엄청난 계산량을 요구한다. 따라서 DP에서는 기존 Bellman equation의 update rule을 iterative하게 전환하였다. 

현재 state의 new value $v_{k+1}(s)$는 transition된 후속 state $s'$의 old value $v_k(s')$로부터 계산된다. 즉, 다시 재귀적으로 계산하지 않고 지금까지 계산된 $v_k(s')$을 그대로 사용하겠다는 것이다. iterative policy evaluation의 각 iteration은 모든 state에 대해 한번에 이러한 update rule을 수행한다. 따라서 iterative policy evaluation을 수행하기 위해서는 일반적으로 old value를 저장한 array와 new value를 계산하기 위한 array가 각각 필요하다. 그러나 실제로는 한개의 array로 old value의 보관과 new value의 계산을 동시에 수행한다. 이때는 new value가 계산되면 기존 old value를 실시간으로 덮어쓴다. 따라서 각 state에 대한 value 계산 시 old value가 참조될 수도 있고, 이미 계산이 완료된 new value가 참조될 수도 있다. 2-array든 1-array 방식이든 수렴성은 보장되며 1-array 방식이 수렴속도가 일반적으로 더 빠르다. 다만 1-array 방식은 state value를 update하는 순서에 따라 수렴 속도가 변한다.

지금까지 DP를 통해 state value $v_\pi$를 계산하는 방법을 알아보았다. 이제 계산된 state value를 통해 policy $\pi$를 개선해보자.

## Policy Improvement

임의의 deterministic policy $\pi$를 따를 때의 state value $v_\pi$가 결정되었다고 가정하자. deterministic policy의 의미는 policy $\pi$의 결과과 action들의 확률 분포가 아닌 action 그 자체인 경우를 말한다. 이 때 어떻게 기존 policy를 개선할 수 있을까? 이는 action value $q_\pi$를 통해 수행할 수 있다. 먼저 action value $q_\pi$를 remind 하자.

$$
\begin{align}
    q_\pi(s,a) &\doteq \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \ \vert \ S_t = s, A_t = a] \\
    &= \sum_{s',r}p(s',r \vert s,a)\Big[r + \gamma v_\pi(s') \Big] \tag{3}
\end{align}
$$

이때 각 state에서 가장 좋은 action value $q_\pi(s, a)$를 가진 action $a$를 선택하면 된다. 이러한 방식을 *greedy* policy라고 하며 new policy $\pi'$은 아래와 같다.

$$
\pi'(s) \doteq \underset{a}{\arg\max} \ q_\pi(s, a) \tag{4}
$$

개선된 정책 $\pi'(s)$는 기존 $\pi(s)$에 의한 action과 같을 수도 있고 다를 수도 있지만 무엇이든 간에 기존 policy 보다는 좋을 것이다. 기존 policy에 의한 value function에 관해 greedy하게 선택하는 과정을 *policy improvement*라고 한다.

만약 new greedy policy $\pi'$이 기존 policy $\pi$와 동일하면 어떨까? 이 경우 기존 policy와 동일한 action을 선택하기 때문에 결국 $v_\pi = v_{\pi'}$이 되며 수식 (4)에 의해 아래와 같은 수식을 만족한다.

$$
v_{\pi'}(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_{\pi'}(S_{t+1}) \ \vert \ S_t = s, A_t = a]
$$

그런데 위 수식은 Bellman optimality equation과 동일하다. 따라서 $v_{\pi'}$은 $v_\ast$이며 $\pi$와 $\pi'$ 모두 optimal policy이다.

## Policy Iteration

*policy iteration*은 policy evaluation과 policy improvement를 번갈아 수행하는 방법이다. 어떤 policy $\pi$를 평가해 $v_\pi$를 구한 뒤 이를 바탕으로 더 나은 policy $\pi'$를 얻는다. 다시 $\pi'$을 평가해 $v_{\pi'}$을 구한 뒤 이를 바탕으로 더 나은 policy $\pi''$를 얻는다. 이러한 과정을 반복하는 것이 바로 policy iteration이다. 아래는 policy iteration을 sequence 형태이다.

$$
\pi_0 \overset{E}{\longrightarrow} v_{\pi_0} \overset{I}{\longrightarrow} \pi_1 \overset{E}{\longrightarrow} v_{\pi_1} \overset{I}{\longrightarrow} \pi_2 \overset{E}{\longrightarrow} \cdots \overset{I}{\longrightarrow} \pi_\ast \overset{E}{\longrightarrow} v_\ast
$$

위 수식에서 $\overset{E}{\longrightarrow}$는 policy evaluation, $\overset{I}{\longrightarrow}$는 policy improvement를 나타낸다. finite MDP는 유한하기 때문에 이러한 프로세스는 유한한 iteration 안에 optimal policy와 optimal value function으로 반드시 수렴한다.

아래는 policy iteration에 대한 알고리즘이다.

> 1. Initialization  
> $V(s) \in \mathbb{R}$ and $\pi(s) \in \mathcal{A}(s)$ arbitrarily for all $s \in \mathcal{S}$  
> <br>
> 2. Policy Evaluation  
> **Loop**:  
> $\quad$ $\Delta \leftarrow 0$  
> $\quad$ **For each** $s \in \mathcal{S}$:  
> $\quad\quad$ $v \leftarrow V(s)$  
> $\quad\quad$ $V(s) \leftarrow \sum_{s',r}p(s',r \vert s, \pi(s))[r + \gamma V(s')]$  
> $\quad\quad$ $\Delta \leftarrow \max(\Delta, \vert v - V(s) \vert)$  
> **until** $\Delta < \theta$ (a small positive number determining the accuracy of estimation)  
> <br>
> 3. Policy Improvement  
> $\textit{policy-stable} \leftarrow \textit{true}$  
> **For each** $s \in \mathcal{S}$:  
> $\quad$ $\textit{old-action} \leftarrow \pi(s)$  
> $\quad$ $\pi(s) \leftarrow \arg\max_a \sum_{s',r}p(s',r \vert s,a)[r + \gamma V(s')]$  
> $\quad$ **If** $\textit{old-action} \neq \pi(s)$, **then** $\textit{policy-stable} \leftarrow \textit{false}$  
> **If** $\textit{policy-stable}$, **then** stop and return $V \approx v_\ast$ and $\pi \approx \pi_\ast$; **else** go to 2

참고로 2. Policy Evaluation에서 state value를 구할 때와 3. Policy Improvement에서 개선된 policy $\pi(s)$를 얻기 위해 action value를 구할 때의 수식이 동일한데 그 이유는 policy $\pi$를 deterministic 하다고 가정했기 때문이다. state value를 구할 때 policy $\pi$를 따를 때의 action value $q_\pi(s, a)$에 대한 expectation을 취하는데 policy $\pi$를 따르는 action $a$의 확률은 1, 나머지 action은 모두 0이기 때문에 $a = \pi(s)$에 대한 action value $q_\pi(s, \pi(s))$가 곧 state value이다.

## Value Iteration

TODO: 내용 추가 예정