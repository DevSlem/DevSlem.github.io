---
title: "n-step Bootstrapping"
tags: [RL, AI]
date: 2022-07-19
last_modified_at: 2022-07-19
sidebar:
    nav: "rl"
---

이 포스트에서는 TD method의 확장된 형태인 $n$-step TD methods를 간략히 소개한다.

## What is $n$-step TD method

*$n$-step TD method*는 1-step TD method와 Monte Carlo (MC) method를 통합한 방법이다. $n$-step TD method는 일종의 스펙트럼으로 양 끝단에 각각 1-step TD와 MC method가 존재한다. 

이전 포스트에서 TD method는 MC method와 Dynamic Programming (DP)의 아이디어를 결합한 방법이라고 소개했었다. $n$-step method 역시 동일하다. 즉, sampling과 bootstrapping을 통해 training이 이루어진다. 다만 1-step TD와의 차이점은 bootstrapping이 이루어지는 time step이 1개가 아니라 여러 개일 뿐이다.

## $n$-step TD Prediction

MC method는 episode를 완전히 고려하기 때문에 전체 reward를 알고 있어 실제 return을 구할 수 있었다.

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1}R_T
$$

$T$는 episode의 last time step이다. 그런데 1-step TD method는 episode 전체가 아니라 sampling된 단 하나의 transition 고려하기 때문에 next reward $R_{t+1}$만 구할 수 있다. 따라서 아직 sampling 되지 않은 time step들의 discounted reward들을 아래와 같이 next state에서의 추정치로 근사해 처리한다.

$$
G_{t:t+1} \doteq R_{t+1} + \gamma V_t(S_{t+1})
$$

$G_{t:t+1}$은 time step $t$에서 $t+1$까지의 transition이 발생했을 때의 target으로 1-step return이라고 한다. 참고로 MC method의 target은 실제 return $G_t$이다. 우리는 위 1-step TD method의 target을 $n$-step으로 확장할 것이다. 먼저 아래 backup diagram을 보자.

![](/assets/images/rl-sutton-n-step-method-backup-diagram.png){: w="60%"}
_Fig 1. Backup diagrams of $n$-step methods.  
(Image source: Sec 7.1 Sutton & Barto (2018).)_  

위 backup diagram을 보면 알 수 있지만 1-step TD 수행 시 실제 reward는 $R_{t+1}$만 획득할 수 있다. 2-step TD 수행 시 $R_{t+1}$과 $R_{t+2}$만 획득할 수 있다. 획득하지 못한 나머지 reward는 기존에 학습된 추정치로 대체한다. 즉, 2-step TD 수행 시 $S_{t+2}$에 위치할 것이기 때문에 $V(S_{t+2})$를 사용한다. 아래는 2-step TD method의 target이다.

$$
G_{t:t+2} \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 V_{t+1}(S_{t+2})
$$

이를 $n$-step TD method로 일반화하면 아래와 같다.

$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})
$$

위 $n$-step TD method의 target을 *$n$-step return*이라고 한다. 정리하면 $n$-step return은 실제 return의 근사치로, $n$ step 이후 잘려진 뒤 $V_{t+n-1}(S_{t+n})$에 의해 보정된 값이다. 이 때 $t + n \geq T$일 경우, episode의 termination을 넘어갔기 때문에 보정된 값은 0이 되어 실제 return이 된다.

$$
G_{t:t+n} \doteq G_t, \quad \text{if } t + n \geq T
$$

$n$-step return은 $n$ step 이후의 $R_{t+n}$과 이전에 계산된 $V_{t+n-1}$이 발견이 되어야만 계산할 수 있다. 따라서 $t+n$ time step 이후에만 $G_{t:t+n}$을 이용할 수 있다. 아래는 $n$-step return을 사용해 state value를 추정하는 알고리즘이다.

$$
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha [G_{t:t+n} - V_{t+n-1}(S_t)]
$$

이 때 $s \neq S_t$인 모든 state의 value는 변하지 않는다. 위 알고리즘이 *$n$-step TD*이다. episode의 첫 $n-1$ step까지는 어떤 변화도 발생하지 않는다는 사실을 꼭 기억하길 바란다.

> ##### $$ \text{Algorithm: } n\text{-step TD for estimating } V \approx v_\pi $$  
> $$ 
> \begin{align*}
> & \text{Input: a policy } \pi \\
> & \text{Algorithm parameters: step size } \alpha \in (0, 1] \text{, a positive integer } n \\
> & \text{Initialize } V(s) \text{ arbitrarily, for all } s \in \mathcal{S} \\
> & \text{All store and access operations (for } S_t \text{ and } R_t \text{) can take their index mode } n+1 \\
> & \\
> & \text{Loop for each episode:} \\
> & \qquad \text{Initialize and store } S_0 \neq \text{terminal} \\
> & \qquad T \leftarrow \infty \\
> & \qquad \text{Loop for } t = 0, 1, 2, \dotso : \\
> & \qquad\qquad \text{If } t < T \text{, then:} \\
> & \qquad\qquad\qquad \text{Take an action according to } \pi(\cdot \vert S_t) \\
> & \qquad\qquad\qquad \text{Observe and store the next reward as } R_{t+1} \text{ and the next state as } S_{t+1} \\
> & \qquad\qquad\qquad \text{If } S_{t+1} \text{ is terminal, then } T \leftarrow t + 1 \\
> & \qquad\qquad \tau \leftarrow t - n + 1 \qquad \text{($\tau$ is the time whose state's estimate is being updated)} \\
> & \qquad\qquad \text{If $\tau \geq 0$:} \\
> & \qquad\qquad\qquad G \leftarrow \textstyle\sum_{i=\tau + 1}^{\min(\tau + n, T)} \gamma^{i - \tau - 1}R_i \\
> & \qquad\qquad\qquad \text{If } \tau + n < T \text{, then: } G \leftarrow G + \gamma^n V(S_{\tau + n}) \qquad (G_{\tau : \tau + n}) \\
> & \qquad\qquad\qquad V(S_\tau) \leftarrow V(S_\tau) + \alpha [G - V(S_\tau)] \\
> & \qquad \text{until } \tau = T - 1
> \end{align*}
> $$
