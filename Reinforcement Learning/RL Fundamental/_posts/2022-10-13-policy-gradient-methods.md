---
title: "Policy Gradient Methods"
tags: [RL, AI, Function Approximation RL, Policy Gradient]
last_modified_at: 2022-10-13
sidebar:
    nav: "rl"
---

드디어 긴 장정 끝에 대망의 마지막 챕터로 왔다. 그 유명한 policy gradient method에 대해 소개하려고 한다. 드디어 나 오늘 여기서 장렬하게 잠든다.

## Introduction

지금까지 우리는 value-based 기반의 방법을 다뤘었다. policy는 추정된 action value에 기반해 action을 선택했다. 이번에는 policy 자체를 매개변수화된 함수로써 학습하는 방법을 볼 것이다. policy parameter vector를 $\mathbf{\theta} \in \mathbb{R}^{d'}$라고 할 때 policy를 parameter $\mathbf{\theta}$가 주어졌을 때 action $a$를 선택하는 확률로써 정의할 수 있다.

$$
\pi(a \vert s, \mathbf{\theta}) = \Pr\{A_t = a \ \vert \ S_t = s, \mathbf{\theta}_t = \mathbf{\theta} \}
$$

policy parameter에 관한 scalar 성능 수치 $J(\mathbf{\theta})$가 있다고 하자. policy를 이 수치에 대한 gradient에 기반해 policy parameter를 학습할 것이다. 이 성능 수치를 maximize하는 것이 목적이며 이에 따라 $J$에 대해 *gradient ascent*를 수행한다.

$$
\mathbf{\theta}_{t+1} + \mathbf{\theta}_t + \alpha \widehat{\nabla J(\mathbf{\theta}_t)}
$$

$\widehat{\nabla J(\mathbf{\theta}_t)} \in \mathbb{R}^{d'}$는 stochastic 추정치로, 이것의 기대값은 성능 수치의 gradient를 근사화한다. 이러한 일반적인 schema를 따르는 모든 방법들을 *policy gradient methods*라고 부른다.

## Policy Approximation and its Advantages

policy gradient method에서는 $\pi(a \vert s, \mathbf{\theta})$가 미분 가능하다. 또한 exploration을 보장하기 위해 일반적으로 policy는 절대 deterministic하지 않으며 즉, stochastic (i.e., $\pi(a \vert s, \mathbf{\theta}) \in (0, 1), \text{ for all } s, a, \mathbf{\theta}$) 하다. policy-based method는 discrete action space 뿐만 아니라 continuous action space에서도 쉽게 적용 가능하다는 엄청난 장점이 있다.

action space가 discrete이고 너무 크지 않다고 하자. 이때 state-action pair에 대한 매개변수화된 수치적인 선호도를 $h(s, a, \mathbf{\theta})$라고 한다면, 아래와 같이 policy $\pi$를 soft-max distribution으로 정의할 수 있다.

$$
\pi(a \vert s, \mathbf{\theta}) \doteq \dfrac{e^{h(s,a,\mathbf{\theta})}}{\sum_b e^{h(s,b,\mathbf{\theta})}}
$$

이러한 방식이 가지는 이점은 크게 아래와 같다.

1. policy가 점점 deterministic policy로 다가가도록 근사화됨
2. 임의의 확률을 가진 action 선택을 가능하게 함
3. policy가 종종 근사화하기에 더 단순한 함수이며 학습 속도가 빠름
4. 사전 지식을 주입하기에 더 좋음

1번의 경우 어떤 state에서의 optimal policy가 deterministic할 때 policy-based method는 deterministic 하도록 근사화하지만, action-value method는 $\epsilon$-greedy policy를 사용할 때 항상 $\epsilon$의 확률로 랜덤하게 action을 선택해야 한다. 

> $\epsilon$을 0으로 설정하면 다른 state에서도 deterministic하게 되므로 절대 바람직 하지 않음
{: .prompt-danger}

2번의 경우 카드게임 같은 확률 게임을 생각해볼 수 있다. 이러한 게임들은 optimal policy가 stochastic하다. 아래 그림을 보면 왜 $\epsilon$-greedy에 비해 훨씬 우월한지 알 수 있다.

![](/assets/images/rl-sutton-ex13.1.png){: w="80%"}

위 그림은 $\epsilon = 0.1$일 때의 상황으로 optimal (약 0.59의 확률)에 비해 획득한 value가 훨씬 낮다.

## The Policy Gradient Theorem

policy 매개변수화에 대한 중요한 이론적인 이점이 하나 더 있다. 연속적인 policy 매개변수화는 action 선택 확률을 학습된 parameter의 함수로써 부드럽게 변화시킨다. 반면 $\epsilon$-greedy는 매우 작은 변화로 maximum action value를 가지는 action이 변경되면 확률이 급격히 변경된다. 따라서 policy-gradient method는 수렴성이 더 강력히 보장된다.

성능 수치 $J(\mathbf{\theta})$는 episodic과 continuing task에서 다르게 정의되긴 하지만 일단은 동등하게 다루겠다. 먼저 episodic case에 대해 다룬자. 성능 수치를 episode의 시작 state에서의 value로 정의하자. 모든 episode는 특정한 state $s_0$에서 시작한다고 할 때 아래와 같이 정의된다.

$$
J(\mathbf{\theta}) \doteq v_{\pi_\mathbf{\theta}}(s_0)
$$

$v_{\pi_\mathbf{\theta}}$는 $\pi_\mathbf{\theta}$에 대한 true value function이다. 여기서는 no discounting ($\gamma = 1$)을 가정한다.

성능 개선을 보장하는 방향으로 policy parameter를 변경해야 한다. 문제는 성능이 action 선택과 state distribution에 의존한다는 점이다. action 선택은 별 문제되지 않지만 state distribution은 다르다. 우리는 일반적으로 state distribution을 모른다. 이것은 environment의 함수이기 때문이다.

*policy gradient theorem*을 통해 이 문제를 쉽게 해결할 수 있다. state distribution에 대한 미분을 포함하지 않고, policy parameter에 관한 성능의 gradient에 대한 analytic 표현을 제공한다. 아래는 episodic case에 대한 policy gradient theorem이다.

$$
\nabla J(\mathbf{\theta}) \propto \sum_s \mu(s) \sum_a q_\pi(s, a) \nabla \pi(a \vert s, \mathbf{\theta})
$$

$\propto$는 "비례한다"의 의미이다. distribution $\mu$는 $\pi$하에서의 on-policy distribution이다.[^1]

이제 policy gradient theorem 증명 과정을 살펴보자. notation의 단순화를 위해 $\pi$는 $\mathbf{\theta}$의 함수이며, 모든 gradient는 $\mathbf{\theta}$에 관한 것임을 암시적으로 나타낸다. 먼저 시작은 state-value function의 gradient를 action-value function에 관해 나타내는 것으로 시작한다.

$$
\begin{align*}
    \nabla v_\pi(s) &= \nabla \bigg[ \sum_a \pi(a \vert s) q_\pi(s,a) \bigg], \quad \text{for all $s \in \mathcal{S}$} \\
    &= \sum_a \Big[ \nabla \pi(a \vert s) q_\pi(s,a) + \pi(a \vert s) \nabla q_\pi(s,a) \Big] \quad \text{(product rule of calculus)} \\
    &= \sum_a \Big[ \nabla \pi(a \vert s) q_\pi(s,a) + \pi(a \vert s) \nabla \sum_{s',r}p(s',r \vert s,a)\big(r + v_\pi(s')\big) \Big] \\
    &= \sum_a \Big[ \nabla \pi(a \vert s) q_\pi(s,a) + \pi(a \vert s) \sum_{s'}p(s' \vert s,a) \nabla v_\pi(s') \Big]
\end{align*}
$$

곧 업데이트...

<!-- 위 수식에서 $\nabla q_\pi(s,a)$를 다시 $v_\pi$에 관해 나타낼 수 있다.[^2] -->



## References

[1] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction; 2nd Edition. 2020.  

## Footnotes

[^1]: DevSlem. [On-policy Prediction with Approximation. The Prediction Objective ($\overline{\text{VE}}$)](../on-policy-prediction-with-approximation/#the-prediction-objective-overlinetextve). [On-policy Control with Approximation. Average Reward: A New Problem Setting for Continuing Tasks. Ergodicity](../on-policy-control-with-approximation/#ergodicity).  
[^2]: DevSlem. [Finite Markov Decision Processes. Bellman Expectation Equation](../finite-markov-decision-processes/#bellman-expectation-equation).