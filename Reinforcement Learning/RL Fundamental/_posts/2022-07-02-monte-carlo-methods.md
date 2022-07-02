---
title: "Monte Carlo Methods in RL"
tags: [RL, AI]
date: 2022-07-03
last_modified_at: 2022-07-03
sidebar:
    nav: "rl"
---

이 포스트에서는 RL에서 environment에 대한 지식을 완전히 알 수 없을 때 experience를 통해 문제를 해결하는 Monte Carlo methods를 소개한다.

## Introduction

*Monte Carlo* (MC) *methods*는 반복된 random sampling을 통해 numerical한 근사해를 얻는 방법으로 일종의 simulation 기법이다. *Reinforcement Learning* (RL)에서 environment에 대한 지식을 완전히 알지 못할 때 MC methods는 굉장히 유용하다. DP에서와 같이 어떤 state에서 next state로의 transition에 대한 정보를 사전에 알 필요 없이 model은 단순히 transition을 sampling하기만 하면 된다. agent는 sampling된 state, action, reward와 같은 *experience*의 sequence를 바탕으로 sample return들에 대해 평균을 냄으로써 RL problem을 해결한다.

MC methods는 일반적으로 experience를 episode 단위로 나누어 적용된다. 따라서 모든 episode는 반드시 terminal state가 존재하는 episodic task여야 한다. MC methods 역시 *Generalized Policy Iteration* (GPI)를 따르며 한 episode가 끝날 때마다 policy evaluation과 policy improvement가 수행되는 step-by-step (online)이 아닌 episode-by-episode 방법이다.

## Monte Carlo Prediction

먼저 value function을 추정하는 policy evaluation 혹은 prediction에 대해 알아보자. state value의 가장 간단한 정의는 **state $s$에서 시작하여 policy $\pi$를 따를 때 얻을 수 있는 expected return**이다.

$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \ \vert \ S_t = s]
$$

Monte Carlo methods는 episode 단위로 수행되기 때문에 먼저 policy $\pi$를 따라 episode를 생성한다. 그 후 episode 내의 experience로부터 위 정의를 그대로 따라 방문된 state에 대한 value function을 추정한다.

같은 episode 내에서 동일한 state가 여러번 방문될 수 있다. 이때 크게 2가지 방법으로 처리한다.

1. first-visit MC method
2. every-visit MC method

*first-visit MC method*는 한 episode 내에서 state $s$를 처음 방문했을때의 return만 그 state의 average return에 반영한다. 반면 *every-visit MC method*는 한 episode 내에서 state $s$를 방문할때마다 그때의 return을 모두 average return에 반영한다. 이 포스트에서는 first-visit MC method만 고려할 계획이다.

MC method를 backup diagram으로 나타내면 아래와 같다. 빨간색 영역이 한 episode이다.

![](/assets/images/rl-mc-backup-diagram.png){: w="43%"}
_Fig 1. MC backup diagram.  
(Image source: Robotic Sea Bass. [An Intuitive Guide to Reinforcement Learning](https://roboticseabass.com/2020/08/02/an-intuitive-guide-to-reinforcement-learning/).)_  

*Dynamic Programming* (DP)에서는 가능한 모든 transition을 고려했다면 MC methods에서는 sampling된 한 episode만 고려하고 있음을 알 수 있다. 또한 DP에서는 one-step transition만을 고려했지만 MC methods에서는 episode가 끝날 때까지의 모든 transition을 고려한다.

## Monte Carlo Estimation of Action Values

[Monte Carlo Prediction](#monte-carlo-prediction)에서 소개한 방법은 value-based 방법이다. 그러나 이 방법은 policy를 개선할 때 큰 문제가 발생하는데 **결국 environment에 대한 지식인 state transition과 관련된 probability distribution을 알아야 한다**.[^1]  policy improvement를 통해 new greedy policy $\pi'$을 획득한다고 할 때 수식은 아래와 같다.

$$
\pi'(s) \doteq \underset{a}{\arg\max} \ q_\pi(s, a)
$$

new policy를 결정하는 방법은 간단하다. 그 state에서의 action value만 알고 있으면 된다. 문제는 action value를 구할 때 발생한다. action value $q_\pi(s, a)$를 구하는 수식은 아래와 같다.

$$
\begin{align}
    q_\pi(s,a) &\doteq \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \ \vert \ S_t = s, A_t = a] \\
    &= \sum_{s',r}p(s',r \vert s,a)\Big[r + \gamma v_\pi(s') \Big]
\end{align}
$$

위 수식을 보면 알 수 있듯이 state value function $v_\pi$를 통해 action value $q_\pi$를 구할 때 environment dynamics $p(s',r \vert s,a)$를 알아야만 한다. policy evaluation에서 random sampling을 통해 environment에 대한 지식 없이 state value $v_\pi(s)$를 추정할 수 있었지만, policy improvement를 수행하기 위해서는 결국 environment에 대해 알고 있어야만 하는 문제가 발생한다.

위 문제는 environment에 대해 알고 있다면 value-based MC methods를 통해서도 해결할 수 있다. 그러나 environment에 대해 알지 못한다면 다른 접근법이 필요한데, state value를 추정하는 것이 아니라 state-action pair에 대한 *action* value 그 자체를 추정하면 이 문제를 해결할 수 있다. action value $q_\pi$의 가장 간단한 정의는 **state $s$에서 policy $\pi$에 따라 action $a$를 선택했을 때 얻을 수 있는 expected return**이다.

$$
q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \ \vert \ S_t = s, A_t = a]
$$

위 정의에 따라 action value 자체를 추정해 획득한 값으로 policy improvement를 수행하면 environment에 대한 정보가 필요 없어진다.

## Monte Carlo Control

이제 Monte Carlo estimation이 optimal policy를 추정하는 policy improvement 혹은 control에 어떻게 적용될 수 있는지 알아보자. 앞서 언급했지만 MC methods 역시 GPI를 따른다. 다만 이제 policy evaluation에서 state value가 아닌 action value $q_\pi$를 추정할 것이다.

![](/assets/images/rl-sutton-gpi-action-value.png){: w="40%"}
_Fig 2. GPI for action value.  
(Image source: Sec 5.3 Sutton & Barto (2017).)_  

위 그림을 sequence로 나타내면 아래와 같다.

$$
\pi_0 \overset{E}{\longrightarrow} q_{\pi_0} \overset{I}{\longrightarrow} \pi_1 \overset{E}{\longrightarrow} q_{\pi_1} \overset{I}{\longrightarrow} \pi_2 \overset{E}{\longrightarrow} \cdots \overset{I}{\longrightarrow} \pi_\ast \overset{E}{\longrightarrow} q_\ast
$$

위에서 $\overset{E}{\longrightarrow}$는 policy evaluation, $\overset{I}{\longrightarrow}$는 policy improvement를 나타낸다. 

policy evaluation은 앞서 [Monte Carlo Prediction](#monte-carlo-prediction)에서 소개한 방식과 동일하다. 현재 policy $\pi$에 따라 state-action pair에 대한 experience를 통해 episode가 생성되고, 이 episode에 대해 관찰된 return들을 바탕으로 action value $q_\pi$를 추정한다. 그 후 계산된 action value $q_\pi$를 바탕으로 episode 내에 방문된 모든 state에 대해 policy가 개선된다. 이를 episode 단위로 반복하다 보면 optimal policy $\pi_\ast$와 optimal action value $q_\ast$를 획득할 수 있다.

다만 위와 같이 수렴하기 위해서는 반드시 모든 state-action pair가 방문된다는 가정이 필요하다. 즉, agent는 environment에 대해 골고루 탐색해야 한다. 이는 RL에서 굉장히 일반적인 *maintaining exploration* 문제이다. action value를 통해 policy evaluation을 효과적으로 수행하기 위해서는 반드시 지속적인 exploration을 보장해야한다. 이 포스트에서는 이 문제에 대한 해결책으로 2가지 방법을 소개한다.

* exploring starts
* $\epsilon$-soft policy

*exploring starts* (ES)는 episode 시작 시 state-action pair $(s, a)$를 stochastic하게 선택하며, 이 때 모든 시작 state-action pair는 반드시 0이 아닌 확률을 가진다. 이 경우 episode의 수가 무한할 경우 모든 state-action pair가 방문됨을 보장할 수 있다.

그러나 위 방법은 특정 state에서 시작해야하는 조건이 있는 경우 유용하지 않다. 이에 대한 대안으로 각 state에서 선택할 모든 action에 대해 0이 아닌 확률을 보장하는 stochastic policy를 고려한다. 대표적으로 *$\epsilon$-soft policy*가 있다. 

이제 각 방법을 바탕으로 한 MC methods 알고리즘을 알아보자.

## Monte Carlo ES

Monte Carlo ES 알고리즘은 아래와 같다.

> $\text{Initialize:}$  
> $\qquad \pi(s) \in \mathcal{A} \text{ (arbitrarily), for all } s \in \mathcal{S}$  
> $\qquad Q(s,a) \in \mathbb{R} \text{ (arbitrarily), for all } s \in \mathcal{S}, \ a \in \mathcal{A}(s)$  
> $\qquad Returns(s,a) \leftarrow \text{empty list, for all } s \in \mathcal{S}, \ a \in \mathcal{A}(s)$  
> 
> **$\textbf{Loop}$**$\text{ forever (for each episode):}$  
> $\qquad \text{Choose } S_0 \in \mathcal{S}, \ A_0 \in \mathcal{A}(S_0) \text{ randomly such that all pairs have probability} > 0$  
> $\qquad \text{Generate an episode from } S_0, A_0, \text{ following } \pi \text{: } S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$  
> $\qquad G \leftarrow 0$  
> **$\qquad \textbf{Loop}$** $\text{ for each step of episode, } t = T-1, T-2, \dots, 0 \text{:}$  
> $\qquad\qquad G \leftarrow \gamma G + R_{t+1}$  
> **$\qquad\qquad \textbf{Unless}$** $\text{ the pair } S_t, A_t \text{ appears in } S_0, A_0, S_1, A_1, \dots, S_{t-1}, A_{t-1} \text{:}$  
> $\qquad\qquad\qquad \text{Append } G \text{ to } Returns(S_t,A_t)$  
> $\qquad\qquad\qquad Q(S_t,A_t) \leftarrow \text{average}(Returns(S_t,A_t))$  
> $\qquad\qquad\qquad \pi(S_t) \leftarrow \arg\max_a Q(S_t,a)$

exploring starts이기 때문에 각 episode의 시작마다 모든 state-action pair의 확률이 0보다 큰 조건 하에 랜덤하게 state-action pair를 선택한다.

## Monte Carlo Control without ES

TODO: #7 Monte Carlo Control without ES 파트 곧 작성될 예정.

[^1]: [Why are state-values alone not sufficient in determining a policy (without a model)?](https://ai.stackexchange.com/questions/22907/why-are-state-values-alone-not-sufficient-in-determining-a-policy-without-a-mod)