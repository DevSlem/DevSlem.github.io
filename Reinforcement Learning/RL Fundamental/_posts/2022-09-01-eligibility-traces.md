---
title: "Eligibility Traces"
tags: [RL, AI, Function Approximation RL]
last_modified_at: 2022-09-01
sidebar:
    nav: "rl"
---

이번 포스트에서는 TD와 Monte Carlo method를 통합 및 일반화하는 eligibility traces에 대해 다룰 것이다.

## Introduction

eligibility traces는 TD와 Monte Carlo (MC) method를 통합 및 일반화하는 방법으로 스펙트럼에 걸쳐 있다. 스펙트럼의 양 끝에는 MC method ($\lambda=1$)와 1-step TD method ($\lambda=0$)가 있다. 또한 eligibility traces는 MC method를 online 학습과 continuing task에서의 학습을 가능하게 한다.

eligibility traces와 비슷한 방법으로 $n$-step TD method가 존재한다.[^1] 그러나 eligibility traces는 $n$-step TD method보다 우아한 알고리즘적 메커니즘을 지니고, 상당한 계산적 이점을 가진다.

eligibility traces는 short-term memory vector인 *eligibility trace* $\mathbf{z}_t \in \mathbb{R}^d$와 동시에 long-term weight vector $\mathbf{w}_t \in \mathbb{R}^d$를 사용한다. 이 둘이 무슨 역할을 하는 지 곧 알아볼 것이다.

eligibility traces가 $n$-step method에 비해 가지는 주요한 계산적 이점은 마지막 $n$개의 feature vector를 저장하는 대신, 단 하나의 trace vector만을 사용한다는 것에서 비롯된다. 또한 $n$-step method는 $n-1$ time step만큼 학습이 지연되고 episode 종료를 포착해야하는 반면, eligibility traces는 학습이 지속적이고 균일하게 수행된다.

MC method와 $n$-step method는 각각 모든 미래 reward와 $n$개의 reward에 기반해 업데이트가 수행된다. 이렇게 업데이트되는 state로부터 앞 혹은 미래를 바라보는 것에 기반하는 방법을 *forward view*라고 부른다. 그러나 eligibility trace를 사용할 경우, 업데이트되는 state로부터 최근 방문했던 state를 향해 뒤 혹은 과거를 바라보는데, 이러한 방법을 *backward view*라고 한다.

이 내용들을 이제 차근차근 알아볼 것이다. 평소처럼 먼저 state value에 대한 prediction을 알아본 뒤, action value와 control로 확장한다.

## The $\lambda$-return

먼저, $n$-step return에 대해 리뷰를 하자. $n$-step return $G_{t:t+n}$은 아래와 같이 $n$개의 discounted reward와 $n$-step에 도달된 state의 discounted 추정치를 더한 값으로 정의된다.

$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n\hat{v}(S_{t+n}, \mathbf{w}_{t+n-1}), \quad 0 \leq t \leq T - n
$$

$\hat{v}(s,\mathbf{w})$는 weight vector $\mathbf{w}$가 주어졌을 때 state $s$에서의 추정치이다. $T$는 episode의 terminal time step이며 $n \geq 1$이다.

우리의 이번 목적인 $\lambda$-return은 $n$-step return을 모든 $n$에 대해 평균을 낸 것으로 TD($\lambda$) 알고리즘에 사용된다. 각 가중치는 $\lambda^{n-1} \text{ (where $\lambda \in [0, 1)$)}$에 비례하며, 가중치의 총합을 1로 설정하기 위해 $1 - \lambda$에 의해 정규화된다. 아래는 $\lambda$-return의 정의이다.

$$
G_t^\lambda \doteq (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1}G_{t:t+n}
$$

위 수식을 아래와 같이 episode 종료 전후로 분리할 수 있다.

$$
G_t^\lambda = (1 - \lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1}G_{t:t+n} + \lambda^{T-t-1}G_t
$$

이 수식은 $\lambda = 1$일 때의 상황을 더 명확히 해준다. $\lambda = 1$일 때 왼쪽 main sum은 0이 되며 기본적인 return $G_t$만 남게 된다. 따라서 $\lambda = 1$일 때 $\lambda$-return은 MC method이다. 반대로 $\lambda = 0$일 때 오직 one-step return $G_{t:t+1}$만 남게 되기 때문에 one-step TD method이다. $\lambda$-return은 $n$-step return과 비교했을 때 MC method와 one-step TD method 사이를 조금 더 부드럽게 이동할 수 있다. 아래는 $\lambda$-return에서 $n$-step return sequence에 가중치를 부여하는 것을 나타낸 backup diagram이다.

![](/assets/images/rl-sutton-td-lambda-backup-diagram.png){: w="70%"}
_Fig 1. The backup diagram for TD($\lambda$).  
(Image source: Sec 12.1 Sutton & Barto (2020).)_  

아래는 각 $n$-step return에 부여되는 가중치의 변화 추이를 나타내는 그림이다. terminal time step 이후의 $n$-step return은 기본적인 return $G_t$이다.

![](/assets/images/rl-sutton-fig12.2.png)
_Fig 2. Weighting given in the $\lambda$-return to each of the $n$-step returns.  
(Image source: Sec 12.1 Sutton & Barto (2020).)_  

이제 $\lambda$-return에 기반한 첫번째 알고리즘을 정의하자. 굉장히 naive한 알고리즘으로 *off-line $\lambda$-return algorithm*이라고 부른다. off-line 알고리즘인 이유는 episode 동안 weight vector가 변하지 않기 때문이다. episode 종료 후에, 전체 off-line update sequence가 아래 일반적인 semi-gradient rule을 따라 수행된다. 이 때 target은 $\lambda$-return $G_t^\lambda$이다.

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \Big[G_t^\lambda - \hat{v}(S_t, \mathbf{w}_t) \Big] \nabla \hat{v}(S_t, \mathbf{w}_t), \quad t = 0, \dots, T - 1
$$

그렇다면 왜 episode 종료까지 기다려야 할까? 이유는 비교적 간단하다. $n$-step TD method는 $n$-step return을 계산하기 위해 $n$번의 transition이 발생할 때까지 기다려야 했다. $\lambda$-return은 기본적으로 $G_{t:t+1}$부터 $G_t$까지 모든 return을 포함한다. $G_t$를 계산하기 위해서는 episode 종료를 기다려야만 한다.

우리가 지금까지 알아본 방법은 forward view이다. time step $t$에서 어떤 state $S_t$를 update할 때 우리는 $t+1, t+2, \dots$와 같이 미래의 보상과 state를 본다. 이 state를 업데이트한 후 다음 state로 넘어가면, 우리는 이전 state를 결코 다시 보지 않는다. 반대로 미래의 것들은 반복적으로 처리된다. 아래는 이러한 forward view의 관계를 나타내는 그림이다.

![](/assets/images/rl-sutton-fig12.4.png)
_Fig 3. The forward view.  
(Image source: Sec 12.1 Sutton & Barto (2020).)_  

## TD($\lambda$)

곧 내용 추가될 예정.

## References

[1] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction; 2nd Edition. 2020.  

## Footnotes

[^1]: DevSlem. [n-step Bootstrapping.](../n-step-bootstrapping/).  