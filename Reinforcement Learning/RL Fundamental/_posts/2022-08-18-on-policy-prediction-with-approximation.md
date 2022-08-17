---
title: "On-policy Prediction with Approximation"
tags: [RL, AI, Function Approximation RL]
date: 2022-08-18
last_modified_at: 2022-08-18
sidebar:
    nav: "rl"
---

이 포스트에서는 reinforcement learning을 수행하는 새로운 방법인 function approximation에 대한 소개와 이를 바탕으로 on-policy method에서 prediction을 수행하는 방법을 소개한다.

## What is Function Approximation and Why needed?

지금까지 알아본 기존 Reinforcement Learning (RL) 방법들은 모두 tabular 기반이었다. tabular 기반 방법들은 작은 state space에서는 잘 작동하지만 매우 큰 state space에 적용할 때는 치명적인 몇가지 문제가 발생한다.

먼저, state space가 매우 클 경우 각 state를 모두 mapping하는 데 **너무 많은 메모리가 요구**된다. 대부분의 RL task들은 가능한 state의 경우의 수가 우주에 존재하는 모든 원자 개수보다 많거나 실수 전체 집합에서 정의되어 무한개이다.

두 번째는 이러한 많은 state space에 대해 **완벽히 계산할 시간이 없다**는 것이다. state space가 클 수록 state에 mapping되는 table은 더 커질 것이고 이에 따라 더 많은 데이터가 필요하고 계산되어야 한다. 이를 수행할만한 시간은 당연히 턱없이 부족하다. 따라서 **대부분의 state는 아예 발견조차 하지 못할 때가 많다**.

더 좋은 학습을 위해선 아직 발견하지 못한 state에 대해서도 적절히 처리할 수 있어야 한다. 이를 위해 **state를 일반화**할 필요성이 있다. 즉, 주요 이슈는 *generalization*이다. 일반화를 달성할 수 있는 방법으로 *function approximation*을 사용한다. function approximation은 말 그대로 함수를 근사하는 방법이다. function approximation은 원래 *supervised learning*에서 사용되던 방법이다. 이를 RL에 적용해 state에 대한 function을 (e.g., value function, policy) 근사하는 것이 앞으로의 주요 과제이다.

function approximation을 RL에 적용할 때 전통적인 supervised learning에서는 나타나지 않던 몇가지 문제가 발생한다. 대표적으로 nonstationarity, bootstrapping, delayed target이 있다. 이 문제를 포함한 function approximation을 RL에 적용할 때 발생하는 여러 문제들을 우리는 앞으로 다룰 것이다. 

## Introduction

이 포스트에서는 function approximation을 적용한 on-policy method에 집중할 것이다. on-policy method는 policy $\pi$로부터 생성된 experience로부터 $\pi$에 대한 함수를 추정하는 방법이었다. 여기서는 $\pi$에 대한 state-value function $v_\pi$를 근사하는 방법을 알아볼 것이다.

기존 tabular 기반 방법과의 가장 주요한 차이점은 추정된 value function이 table이 아닌 **weight vector $\mathbf{w} \in \mathbb{R}^d$로 구성된 매개변수화된 함수로써 표현**된다는 것이다. 따라서 어떤 weight vector $\mathbf{w}$가 주어졌을 때 state $s$에 대한 근사값을 $\hat{v}(s,\mathbf{w}) \approx v_\pi(s)$로 나타낸다. 요즘 대부분의 RL은 weight vector $\mathbf{w}$에 주로 신경망 매개변수를 사용한다. weight의 개수 ($\mathbf{w}$의 차원)는 거의 대부분의 경우 state의 개수보다 훨씬 작다. 즉, $d \ll \vert \mathcal{S} \vert$이다.

weight 하나를 변경하면 많은 state에 대한 추정치도 변하게 된다. 결과적으로 **어떤 state 하나가 업데이트되면 다른 수많은 state의 value에도 영향을 미친다**. 이를 통해 generalization이 가능해진다. 발견한 state에 대해 업데이트하면 아직 발견하지 못한 state의 value도 변경되기 때문이다. 그러나 다른 수많은 state의 value가 변경되기 떄문에 관리하고 이해하는게 어렵다는 점도 있다.

## Value-function Approximation

value function을 업데이트하기 위해서는 이에 대한 target이 필요하다. **target은 추정치가 다가가려는 지향점**으로 생각할 수 있다. 우리는 앞선 여러 tabular 기반 방법에서 각 방법에 대한 target들을 알아봤었다. state $s$에서의 *update target*을 $u$라고 할 때 update를 아래와 같이 표현한다.

$$
s \mapsto u
$$

Monte Carlo (MC) update는 $S_t \mapsto G_t$, TD(0) update는 $S_t \mapsto R_{t+1} + \gamma \hat{v}(S_{t+1},\mathbf{w}_ t)$, $n$-step TD update는 $S_t \mapsto G_{t:t+n}$이다. 또한 Dynamic Programming (DP)의 policy-evaluation update는 $s \mapsto \mathbb{E}_ \pi [R_{t+1} + \gamma \hat{v}(S_{t+1},\mathbf{w}_t) \ \vert \ S_t=s]$ 이다.

이러한 update를 학습 예제로 사용함으로써 value prediction에 대한 function approximation을 수행할 것이다. 각 update를 학습 예제로 간주하는 것은 이미 존재하는 여러 function approximation 방법을 사용할 수 있게 해준다. 하지만 모든 function approximation이 reinforcement learning에 적합하지는 않다. 대부분의 신경망 기반 방법들은 static한 여러 개의 training example로 구성된 training set을 가정한다. reinforcement learning에서는 **agent가 environment와 상호작용하면서 online으로 학습하는 것이 중요**하다. 따라서 점진적으로 획득한 data로부터 효과적으로 학습할 수 있는 방법이 필요하다. 게다가 reinforcement learning에서는 일반적으로 function approximation 방법이 **nonstationary한 target function을 (시간이 지남에 따라 target function이 변함) 다룰 수 있어야** 한다. 일반적인 supervised learning은 target인 정답 레이블이 고정되어 있기 때문에 이러한 문제가 발생하지 않는다.

## The Prediction Objective ($\overline{\text{VE}}$)

곧 내용 추가.
