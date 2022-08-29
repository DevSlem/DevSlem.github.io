---
title: "Off-policy Methods with Approximation"
tags: [RL, AI, Function Approximation RL]
date: 2022-08-23
last_modified_at: 2022-08-25
sidebar:
    nav: "rl"
---

이 포스트에서는 on-policy function approximation을 off-policy로의 확장과 이로 인해 발생하는 문제들에 대해 다룰 것이다.

## Introduction

off-policy method는 *behavior policy* $b$에 의해 생성된 experience로부터 target policy $\pi$에 대한 value function을 학습하는 방법이다. 반대로 on-policy method는 behavior policy와 target policy가 동일한 방법이다. off-policy method에서는 일반적으로 target policy $\pi$는 greedy policy이고, behavior policy $b$는 좀 더 효과적으로 exploration을 수행할 수 있는 policy (e.g., $\epsilon$-greedy policy) 이다.

off-policy method의 문제는 크게 두가지로 나뉜다. 첫 번째는 update target (target policy 아님)과 관련된 이슈로, 이전 tabular off-policy method에서 다뤘었던 importance sampling과 관련되어 있다. 두 번째는 update distribution과 관련된 이슈로, update distribution이 on-policy distribution을 따르지 않는 다는 점이다. on-policy distribution은 semi-gradient method의 안정성에 있어 중요하다. 이 두 이슈에 대해 이번 포스트에서 다룰 것이다.

## Semi-gradient Methods

이 섹션에서는 off-policy method를 간략히 semi-gradient method로 확장한다. 이 확장된 방법은 off-policy method의 첫 번째 이슈에 대해서만 다루며, 수렴이 안되어 발산할 때도 있다.

$n$-step tabular method에서 사용했던 방법을 단순히 function approximation을 사용한 weight vector $\mathbf{w}$에 대한 update로 변경한다. off-policy method는 target policy와 behavior policy의 distribution이 다르기 때문에 importance sampling 기법을 사용한다.[^1] 그 중 off-policy TD(0)와 같은 알고리즘은 per-step importance sampling ratio를 사용한다.

$$
\rho_t \doteq \rho_{t:t} = \dfrac{\pi(A_t \vert S_t)}{b(A_t \vert S_t)}
$$

아래는 off-policy semi-gradient TD(0)로 [semi-gradient on-policy TD(0)](../on-policy-prediction-with-approximation/#textalgorithm-semi-gradient-td0-for-estimating--hatv-approx-v_pi)에 $\rho_t$만 추가되었다.

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \rho_t \delta_t \nabla \hat{v}(S_t, \mathbf{w}_t)
$$

$\delta_t$는 TD error로 아래는 각각 episodic, discounted setting과 average reward를 사용하는 continuing, undiscounted setting에서의 TD error이다.

$$
\begin{align*}
    & \delta_t \doteq R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) \text{, or} \tag{episodic} \\
    & \delta_t \doteq R_{t+1} + \bar{R}_t + \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) \tag{continuing}
\end{align*}
$$

$\bar{R}_t$는 average reward의 추정치이다.[^2]

action value에 대해서도 쉽게 전환할 수 있다. 아래는 off-policy semi-gradient Expected Sarsa이다.

$$
\begin{align*}
    & \mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \nabla \hat{q}(S_t, A_t, \mathbf{w_t}) \text{, with} \\ \\
    & \delta_t \doteq R_{t+1} + \gamma \sum_a \pi(a \vert S_{t+1}) \hat{q}(S_{t+1}, a, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t) \text{, or} \tag{episodic} \\
    & \delta_t \doteq R_{t+1} - \bar{R}_t + \sum_a \pi(a \vert S_{t+1}) \hat{q}(S_{t+1}, a, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t) \tag{continuing}
\end{align*}
$$

이 알고리즘은 importance sampling을 사용하지 않는다. tabular case에서는 action value에 대한 1-step TD method에 importance sampling을 사용하지 않는 이유가 명확했다.[^3] 그러나 function approximation에서는 명확하지 않다. 그 이유를 Sutton 책에서는 아래와 같이 설명한다. (솔직히 내가 이해 못했음)

> With function approximation it is less clear because we might want to weight different state-action pairs differently once they all contribute to the same overall approximation.[^4]

아래는 off-policy semi-gradient Sarsa의 $n$-step version이다.

$$
\begin{align*}
    & \mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \rho_{t+1} \cdots \rho_{t+n} [G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1})] \nabla \hat{q}(S_t,A_t,\mathbf{w}_{t+n-1}) \text{, with} \\ \\
    & G_{t:t+n} \doteq R_{t+1} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}) \text{, or} \tag{episodic} \\
    & G_{t:t+n} \doteq R_{t+1} - \bar{R}_t + \cdots + R_{t+n} - \bar{R}_{t+n-1} + \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}) \tag{continuing}
\end{align*}
$$

위 첫 번째 수식에서 $\rho_k$는 $k \geq T$ ($T$는 episode의 terminal step) 일 때 $1$이다. $t + n \geq T$일 때 $G_{t:t+n} \doteq G_t$이다.

> 세 번째 수식에서 [on-policy semi-gradient $n$-step Sarsa](../on-policy-control-with-approximation/#differential-semi-gradient-n-step-sarsa)의 경우 average reward가 오직 $\bar{R}_{t+n-1}$이었는데 여기서는 다르다. off-policy 여서 다른 건지, 책이 오류인 건지 모르겠다.
{: .prompt-warning}

## References

[1] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction; 2nd Edition. 2020.  

## Footnotes

[^1]: DevSlem. [Monte Carlo Methods in RL. Importance Sampling](../monte-carlo-methods/#importance-sampling).  
[^2]: DevSlem. [On-policy Control with Approximation. Average Reward: A New Problem Setting for Continuing Tasks. Conversion to Average Reward Setting](../on-policy-control-with-approximation/#conversion-to-average-reward-setting).  
[^3]: DevSlem. [n-step Bootstrapping. $n$-step Off-policy Learning](../n-step-bootstrapping/#n-step-off-policy-learning).
[^4]: Reinforcement Learning: An Introduction; 2nd Edition. 2020. Sec. 11.1, p.259.