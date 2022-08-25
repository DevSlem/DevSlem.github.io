---
title: "On-policy Control with Approximation"
tags: [RL, AI, Function Approximation RL]
date: 2022-08-23
last_modified_at: 2022-08-25
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
> & \textstyle \text{Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)} \\
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
> & \textstyle \text{Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)} \\
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

### Average Reward

continuing task에 적용하기 위해 새롭게 도입할 setting은 *average reward* setting이다. 이 setting은 discounted reward가 존재하지 않으며 즉각적인 reward와 delayed reward를 동일하게 취급한다. 이 setting에서 policy $\pi$에 대한 quality는 *average reward* $r(\pi)$에 의해 정의된다.

$$
\begin{align*}
    r(\pi) &\doteq \lim_{h \rightarrow \infty} \dfrac{1}{h} \sum_{t=1}^h \mathbb{E}[R_t \ \vert \ S_0, A_{0:t-1} \sim \pi] \\
    &= \lim_{t \rightarrow \infty} \mathbb{E}[R_t \ \vert \ S_0, A_{0:t-1} \sim \pi] \\
    &= \sum_s \mu_\pi(s) \sum_a \pi(a \vert s) \sum_{s',r} p(s',r \vert s,a)r
\end{align*}
$$

기대값은 시작 state $S_0$와 policy $\pi$에 따라 선택된 subsequent action $A_0, A_1, \dots, A_{t-1}$을 조건으로 한다. MDP가 *ergodic*하다면 위 두번째와 세번째 수식이 성립한다. ergodic하다는게 무슨 의미일까? 이에 대해 알아보자.

### Ergodicity

ergodicity는 시간이 충분할 때 선택된 action들에 상관없이 모든 state가 방문됨을 의미한다. 따라서 $S_0$에 독립적인 steady-state distribution $\mu_\pi(s) \doteq \lim_{t \rightarrow \infty} \Pr \\{ S_t=s \ \vert \ A_{0:t-1} \sim \pi \\}$가 존재한다. distribution $\mu_\pi(s)$가 존재한다는 것은, 모든 state에 대해 어떤 action들이 선택되었든 간에 어떤 state $s$에 방문할 확률이 수렴했음을 의미이다. 

위 정의에 따라 ergodic MDP에서는 시작 state와 초기에 결정된 action들은 오직 일시적인 효과만 가진다. 장기적 관점에서는 어떤 state에 방문할 확률은 오직 policy와 MDP transition probability에 의해 결정된다. ergodicity는 $r(\pi)$의 수렴에 대한 충분조건이지만 필요조건은 아니다.

steady-state distribution $\mu_\pi$ 하에서 policy $\pi$에 따라 action을 선택할 때, 어떤 state $s'$이 방문될 확률 $\mu_\pi(s')$에 대해 아래와 같은 수식이 성립한다.

$$
\sum_s \mu_\pi(s) \sum_a \pi(a \vert s)p(s' \vert s,a) = \mu_\pi(s')
$$

위 수식이 성립하는 이유는 앞서 언급했듯이 어떤 state에 방문할 확률은 오직 policy $\pi$와 transition probability $p$에 의해 결정되기 때문임을 직관적으로 알 수 있다.

### Conversion to Average Reward Setting

average reward setting에서 return은 reward와 average reward의 차이의 관점에서 정의된다.

$$
G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots
$$

이 return을 *differential return*이라고 하며, 이에 대응되는 value function을 *differential value function*이라고 한다. differential value function은 새로운 return만 사용할 뿐, 기존 value function과 동일한 관점에서 정의된다. 즉, differential value function $v_\pi(s) \doteq \mathbb{E}_ \pi[G_t \vert S_t=s]$이고, $q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t \vert S_t=s, A_t=a]$이다. differential value function 역시 bellman equation을 가지나 약간의 차이가 있다. 먼저 discount rate $\gamma$를 제거하고, 모든 reward를 reward와 실제 average reward와의 차이로 대체한다.

$$
\begin{align*}
    & v_\pi(s) = \sum_a \pi(a \vert s) \sum_{r,s'} p(s',r \vert s,a)\Big[r - r(\pi) + v_\pi(s')\Big] \\
    & q_\pi(s,a) = \sum_{r,s'}p(s',r \vert s,a)\Big[r - r(\pi) + \sum_{a'} \pi(a' \vert s')q_\pi(s', a')\Big] \\
    & v_\ast(s) = \max_a \sum_{r,s'} p(s',r \vert s,a)\Big[r - \max_\pi r(\pi) + v_\ast(s')\Big] \\
    & q_\ast(s,a) = \sum_{r,s'} p(s',r \vert s,a) \Big[r - \max_\pi r(\pi) + \max_{a'} q_\ast(s',a')\Big]
\end{align*}
$$

또한 TD error에 대한 differential한 형식 역시 정의할 수 있다.

$$
\begin{align*}
    & \delta_t \doteq R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) \\
    & \delta_t \doteq R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)
\end{align*}
$$

$\bar{R}_t$는 average reward $r(\pi)$의 time step $t$에서의 추정치이다. 이 정의들을 통해 기존 대부분의 알고리즘과 이론적 결과를 특별한 변화 없이 average reward setting으로 전환할 수 있다. 예를 들어 semi-gradient Sarsa의 average reward version은 단순히 TD error을 differential TD error로 전환하기만 하면 된다.

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \nabla \hat{q}(S_t, A_t, \mathbf{w}_t)
$$

$\delta_t$는 action value에 대한 differential TD error이다. differential semi-gradient Sarsa에 대한 전체 알고리즘은 아래 박스와 같다.

> ##### $\text{Algorithm: Differential semi-gradient Sarsa for estimating } \hat{q} \approx q_\ast$  
> $$
> \begin{align*}
> & \textstyle \text{Input: a differentiable action-value function parameterization } \hat{q} : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \rightarrow \mathbb{R} \\
> & \textstyle \text{Algorithm parameters: step size $\alpha, \beta > 0$, small $\epsilon > 0$} \\
> & \textstyle \text{Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w}=\mathbf{0}$)} \\
> & \textstyle \text{Initialize average reward estimate $\bar{R} \in \mathbb{R}$ arbitrarily (e.g., $\bar{R} = 0$)} \\
> \\
> & \textstyle \text{Initialize state $S$, and action $A$} \\
> & \textstyle \text{Loop for each step:} \\
> & \textstyle \qquad \text{Take action $A$, observe $R, S'$} \\
> & \textstyle \qquad \text{Choose $A'$ as a function of $\hat{q}(S', \cdot, \mathbf{w})$ (e.g., $\epsilon$-greedy)} \\
> & \textstyle \qquad \delta \leftarrow R - \bar{R} + \hat{q}(S',A',\mathbf{w}) - \hat{q}(S,A,\mathbf{w}) \\
> & \textstyle \qquad \bar{R} \leftarrow \bar{R} + \beta \delta \\
> & \textstyle \qquad \mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \nabla \hat{q}(S,A,\mathbf{w}) \\
> & \textstyle \qquad S \leftarrow S' \\
> & \textstyle \qquad A \leftarrow A' \\
> \end{align*}
> $$

위 알고리즘은 differential value가 아닌 differential value에 임의의 offset이 더해진 값으로 수렴한다는 이슈가 있다. 그러나 bellman equation이나 TD error는 모든 값이 같은 양만큼 이동하더라도 (즉, 같은 offset이 더해지더라도) 영향을 받지 않는다. 따라서 실제로는 문제가 되지 않는다.

## Deprecating the Discounted Setting

곧 추가될 예정.

## Differential Semi-gradient $n$-step Sarsa

$n$-step return을 function approximation이 사용된 differential 형식으로 아래와 같이 바꿀 수 있다.

$$
G_{t:t+n} \doteq R_{t+1} - \bar{R}_{t+n-1} + \cdots + R_{t+n} - \bar{R}_{t+n-1} + \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1})
$$

$\bar{R}$는 $r(\pi)$의 추정치이며, $n \geq 1$, $t+n < T$이다. $t+n \geq T$이면 $G_{t:t+n} \doteq G_t$이다. differential $n$-step TD error는 아래와 같다.

$$
\delta_t \doteq G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w})
$$

아래 박스는 전체 알고리즘이다.

> ##### $\text{Algorithm: Differential semi-gradient $n$-step Sarsa for estimating $\hat{q} \approx q_\pi$ or $q_\ast$}$  
> $$
> \begin{align*}
> & \textstyle \text{Input: a differentiable function $\hat{q} : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \rightarrow \mathbb{R}$, a policy $\pi$} \\
> & \textstyle \text{Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)} \\
> & \textstyle \text{Initialize average-reward estimate $\bar{R} \in \mathbb{R}$ arbitrarily (e.g., $\bar{R} = 0$)} \\
> & \textstyle \text{Algorithm parameters: step size $\alpha, \beta > 0$, small $\epsilon > 0$, a positive integer $n$} \\
> & \textstyle \text{All store and access operations $(S_t,A_t,R_t)$ can take their index mode $n+1$} \\
> \\
> & \textstyle \text{Initialize and store $S_0$, and $A_0$} \\
> & \textstyle \text{Loop for each step, } t = 0, 1, 2, \ldots : \\
> & \textstyle \qquad \text{Take action $A_t$} \\
> & \textstyle \qquad \text{Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$} \\
> & \textstyle \qquad \text{Select and store an action $A_{t+1} \sim \pi(\cdot \vert S_{t+1})$, or $\epsilon$-greedy wrt $\hat{q}(S_{t+1}, \cdot, \mathbf{w})$} \\
> & \textstyle \qquad \tau \leftarrow t - n + 1 \qquad \text{($\tau$ is the time whose estimate is being updated)} \\
> & \textstyle \qquad \text{If $\tau \geq 0$:}  \\
> & \textstyle \qquad\qquad \delta \leftarrow \sum_{i=\tau+1}^{\tau+n} (R_i - \bar{R}) + \hat{q}(S_{\tau+n}, A_{\tau+n}, \mathbf{w}) - \hat{q}(S_\tau, A_\tau, \mathbf{w}) \\
> & \textstyle \qquad\qquad \bar{R} \leftarrow \bar{R} + \beta \delta \\
> & \textstyle \qquad\qquad \mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \nabla \hat{q}(S_\tau, A_\tau, \mathbf{w}) \\
> \end{align*}
> $$

## References

[1] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction; 2nd Edition. 2020.  

## Footnotes

[^1]: DevSlem. [Finite Markov Decision Processes. Return](../finite-markov-decision-processes/#return).