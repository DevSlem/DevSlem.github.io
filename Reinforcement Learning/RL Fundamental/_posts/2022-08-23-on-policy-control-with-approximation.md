---
title: "On-policy Control with Approximation"
tags: [RL, AI, Function Approximation RL]
date: 2022-08-23
last_modified_at: 2022-08-23
sidebar:
    nav: "rl"
---

이 포스트에서는 function approximation을 사용한 prediction을 control로 확장할 것이다. 이를 위해 state-value function이 아닌 action-value function을 추정한다. 그 후 on-policy GPI의 일반적인 패턴을 따라 학습을 진행하는 방법을 알아본다. 이 포스트에서는 특히 semi-gradient TD(0)의 가장 기본적인 확장인 semi-gradient Sarsa 알고리즘을 다룰 것이다.

## Episodic Semi-gradient Control

function approximation을 통해 매개변수화된 action-value function은 $\hat{q}(s,a,\mathbf{w}) \approx q_\pi(s,a)$이며, 이 때 $\mathbf{w} \in \mathbb{R}^d$는 $d$차원 weight 벡터이다. 또한 $S_t \mapsto U_t$가 아닌 $S_t, A_t \mapsto U_t$ 형식의 training example을 고려한다. $U_t$는 $q_\pi(S_t,A_t)$의 어떤 근사값이든 가능하다. 이를 바탕으로 action-value prediction에 대한 일반적인 gradient-descent update는 아래와 같다.

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \Big[ U_t - \hat{q}(S_t,A_t,\mathbf{w}_t) \Big] \nabla \hat{q}(S_t,A_t,\mathbf{w}_t)
$$

위 update rule을 바탕으로 한 one-step Sarsa의 update rule은 아래와 같다.

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \Big[ R_{t+1} + \gamma \hat{q}(S_{t+1},A_{t+1},\mathbf{w}_t) - \hat{q}(S_t,A_t,\mathbf{w}_t) \Big] \nabla \hat{q}(S_t,A_t,\mathbf{w}_t)
$$

위 방법을 *episodic semi-gradient one-step Sarsa*라고 부른다. continuing task에는 다른 방법이 사용된다.

이제 control을 수행하자. 여기서는 continuous action이나 매우 많은 discrete action에 대해서는 다루지 않는다. 이는 아직까지 연구중인 굉장히 어려운 문제이기 때문이다. action이 discrete하고 그리 많지 않다면 $\epsilon$-greedy policy와 같은 방법을 통해 policy improvement를 수행할 수 있다. 아래 박스는 전체 알고리즘이다.

> ##### $\text{Algorithm: Episodic Semi-gradient Sarsa for Estimating } \hat{q} \approx q_\ast$  
> $$
> \begin{align*}
> & \textstyle \text{Input: a differentiable action-value function parameterization } \hat{q} : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \rightarrow \mathbb{R} \\
> & \textstyle \text{Algorithm parameters: step size $\alpha > 0$, small $\epsilon > 0$} \\
> & \textstyle \text{Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = 0$)} \\
> \\
> & \textstyle \text{Loop for each episode:} \\
> & \textstyle \qquad S,A \leftarrow \text{initial state and action of episode (e.g., $\epsilon$-greedy)} \\
> & \textstyle \qquad \text{Loop for each step of episode:} \\
> & \textstyle \qquad\qquad \text{Take action $A$, observe } R, S' \\
> & \textstyle \qquad\qquad \text{If $S'$ is terminal:} \\
> & \textstyle \qquad\qquad\qquad \mathbf{w} \leftarrow \mathbf{w} + \alpha \big[R - \hat{q}(S,A,\mathbf{w}) \big] \nabla \hat{q}(S,A,\mathbf{w}) \\
> & \textstyle \qquad\qquad\qquad \text{Go to next episode} \\
> & \textstyle \qquad\qquad \text{Choose $A'$ as a function of $\hat{q}(S',\cdot,\mathbf{w})$ (e.g., $\epsilon$-greedy)} \\
> & \textstyle \qquad\qquad \mathbf{w} \leftarrow \mathbf{w} + \alpha \big[R + \gamma \hat{q}(S',A',\mathbf{w}) - \hat{q}(S,A,\mathbf{w}) \big] \nabla \hat{q}(S,A,\mathbf{w}) \\
> & \textstyle \qquad\qquad S \leftarrow S' \\
> & \textstyle \qquad\qquad A \leftarrow A' \\
> \end{align*}
> $$

## Semi-gradient $n$-step Sarsa

episodic semi-gradient Sarsa의 $n$-step version이다. 아래는 update target에 사용되는 $n$-step return으로 function approximation 형식으로 정의되었다.

$$
G_{t:{t+n}} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \quad t + n < T
$$

$t+n \geq T$일 때 $G_{t:t+n} \doteq G_t$이다. $n$-step update rule은 아래와 같다.

$$
\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \Big[ G_{t:t+n} - \hat{q}(S_t,A_t,\mathbf{w}_{t+n-1}) \Big] \nabla \hat{q}(S_t,A_t,\mathbf{w}_{t+n-1}), \quad 0 \leq t < T
$$

아래 박스는 전체 알고리즘이다.

> ##### $\text{Algorithm: Episodic semi-gradient $n$-step Sarsa for estimating } \hat{q} \approx q_\ast \text{ or } q_\pi$  
> $$
> \begin{align*}
> & \textstyle \text{Input: a differentiable action-value function parameterization } \hat{q}: \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \rightarrow \mathbb{R} \\
> & \textstyle \text{Input: a policy $\pi$ (if estimating $q_\pi$)} \\
> & \textstyle \text{Algorithm parameters: step size $\alpha > 0$, small $\epsilon > 0$, a positive integer $n$} \\
> & \textstyle \text{Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = 0$)} \\
> & \textstyle \text{All store and access operations $(S_t, A_t, R_t)$ can take their index mod $n+1$} \\
> \\
> & \textstyle \text{Loop for each episode:} \\
> & \textstyle \qquad \text{Initialize and store $S_0 \neq$ terminal} \\
> & \textstyle \qquad \text{Select and store an action $A_0 \sim \pi(\cdot \vert S_0)$ or $\epsilon$-greedy wrt $\hat{q}(S_0,\cdot,\mathbf{w})$} \\
> & \textstyle \qquad T \leftarrow \infty \\
> & \textstyle \qquad \text{Loop for } t = 0, 1, 2, \ldots : \\
> & \textstyle \qquad\vert\qquad \text{If $t < T$, then:} \\
> & \textstyle \qquad\vert\qquad\qquad \text{Take action } A_t \\
> & \textstyle \qquad\vert\qquad\qquad \text{Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$} \\
> & \textstyle \qquad\vert\qquad\qquad \text{If $S_{t+1}$ is terminal, then:} \\
> & \textstyle \qquad\vert\qquad\qquad\qquad T \leftarrow t + 1 \\
> & \textstyle \qquad\vert\qquad\qquad \text{else:} \\
> & \textstyle \qquad\vert\qquad\qquad\qquad \text{Select and store $A_{t+1} \sim \pi(\cdot \vert S_{t+1})$ or $\epsilon$-greedy wrt $\hat{q}(S_{t+1},\cdot,\mathbf{w})$} \\
> & \textstyle \qquad\vert\qquad \tau \leftarrow t - n + 1 \qquad \text{($\tau$ is the time whose estimate is being updated)} \\
> & \textstyle \qquad\vert\qquad \text{If $\tau \geq 0$:} \\
> & \textstyle \qquad\vert\qquad\qquad G \leftarrow \sum_{i = \tau + 1}^{\min(\tau + n, T)} \gamma^{i - \tau - 1}R_i \\
> & \textstyle \qquad\vert\qquad\qquad \text{If $\tau + n < T$, then } G \leftarrow G + \gamma^n \hat{q}(S_{\tau + n}, A_{\tau + n}, \mathbf{w}) \qquad (G_{\tau : \tau + n}) \\
> & \textstyle \qquad\vert\qquad\qquad \mathbf{w} \leftarrow \mathbf{w} + \alpha \big[G - \hat{q}(S_\tau, A_\tau, \mathbf{w}) \big] \nabla \hat{q}(S_\tau, A_\tau, \mathbf{w}) \\
> & \textstyle \qquad \text{Until } \tau = T - 1 \\
> \end{align*}
> $$

## Average Reward: A New Problem Setting for Continuing Tasks

우리가 그동안 다뤘던 setting은 discounted setting으로 delayed reward에 discount를 적용하는 setting이다. 이 setting은 function approximation을 continuing task에서 사용할 때 문제가 발생한다. 이로 인해 새로운 setting을 도입할 필요가 있다. 사실 tabular 방법을 사용할 때는 오히려 discounted setting이 효과적이다.[^1]

continuing task에 적용하기 위해 새롭게 도입할 setting은 *average reward* setting이다. 이 setting은 discounted reward가 존재하지 않으며 즉각적인 reward와 delayed reward를 동일하게 취급한다. 이 setting에서 policy $\pi$에 대한 quality는 *average reward* $r(\pi)$에 의해 정의된다.

$$
\begin{align*}
    r(\pi) &\doteq \lim_{h \rightarrow \infty} \dfrac{1}{h} \sum_{t=1}^h \mathbb{E}[R_t \ \vert \ S_0, A_{0:t-1} \sim \pi] \\
    &= \lim_{t \rightarrow \infty} \mathbb{E}[R_t \ \vert \ S_0, A_{0:t-1} \sim \pi] \\
    &= \sum_s \mu_\pi(s) \sum_a \pi(a \vert s) \sum_{s',r} p(s',r \vert s,a)r
\end{align*}
$$

기대값은 초기 state $S_0$와 policy $\pi$에 따라 선택된 subsequent action $A_0, A_1, \dots, A_{t-1}$에 의해 결정된다. steady-state distribution $\mu_\pi(s) \doteq \lim_{t \rightarrow \infty} \Pr \{ S_t=s \ \vert \ A_{0:t-1} \sim \pi \}$가 존재하며 $S_0$에 독립적일 때 MDP는 *ergodic*하며, 이 때 위 두번째와 세번째 수식이 성립한다.

ergodic MDP는 ...

## References

[1] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction; 2nd Edition. 2020.  

## Footnotes

[^1]: DevSlem. [Finite Markov Decision Processes. Return](../finite-markov-decision-processes/#return).