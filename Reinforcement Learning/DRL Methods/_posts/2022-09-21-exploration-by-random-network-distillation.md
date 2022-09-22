---
title: "Exploration by Random Network Distillation"
tags:
    - [RL, AI, DRL]
last_modified_at: 2022-09-21
---

이 포스트에서는 exploration을 쉽고 효과적으로 수행할 수 있는 방법인 Exploration by Random Network Distillation 논문을 소개한다.

## Abstract

* exploration bonus는 observation feature의 예측 error임
* 고정 랜덤 초기화 신경망이 사용됨
* extrinsic reward와 intrinsic reward를 유연하게 결합
* Montezuma's Revenge에서 SOTA를 달성

## Introduction

Reinforcement Learning (RL) method는 dense reward 환경에서는 잘 작동한다. dense reward 환경은 랜덤한 action sequence로부터 쉽게 reward를 찾을 수 있는 환경을 의미한다. 반대로 reward를 찾기 어려운 spare reward 환경에서는 학습에 종종 실패한다.

dense reward 환경을 모델링하는 것은 어려우며 부적절하다. dense reward 환경은 일종의 cheating 행위가 될 수도 있다. 따라서 spare reward 환경에서 적절히 환경을 탐색할 수 있는 방법이 필요하다.

이 논문에서 제시한 방법의 특징은 아래와 같다.

* 간단한 구현
* high-dimensional observation에서도 잘 작동
* 어떤 policy optimization 알고리즘에도 적용 가능
* experience batch에 대한 신경망의 단일 forward pass만 요구하기 때문에 효율적

신경망은 학습된 예제와 비슷한 것들에 대해 상당히 낮은 prediction error를 가지는 경향이 있다. 이러한 특징을 기반으로 새로운 experience의 novelty를 정량화함으로써 exploration bonus를 정의할 수 있다.

prediction error를 최대화하려는 agent는 대체로 확률적 transition을 추구하는 경향이 있다. 가장 대표적인 예시가 TV noise이다. TV noise는 계속 확률적으로 변하기 때문에 agent는 항상 새롭다고 느끼게 된다. 그 결과 쓸모없는 noise로 가득찬 TV 화면만 계속 쳐다보게 된다.

이 논문은 위와 같은 문제를 입력에 대한 결정론적 함수를 사용한 exploration bonus를 정의함으로써 해결한다. 이 결정론적 함수는 observation에 대한 고정 랜덤 초기화 신경망이다.

exploration bonus를 extrinsic reward와 결합하기 위해 PPO algorithm을 두 reward stream에 대한 두 개의 state value function을 사용하도록 변형한다. 이를 통해 각 reward에 대해 서로 다른 discount rate 적용이 가능해지며, episodic과 non-episodic return을 결합할 수 있게 해준다.

## Exploration Bonus

환경으로부터 획득하는 reward를 $e_t$라고 하자. 이 때 $e_t$는 sparse하다. exploration bonus $i_t$는 agent가 spare reward 환경을 적절히 탐색하도록 돕는 역할을 한다. 최종적으로 reward function은 $r_t = e_t + i_t$로 정의된다. agent가 새로운 state를 탐색하도록 돕기 위해서는 당연히 자주 방문했던 state보다 새로운 state에서 $i_t$가 높아야 할 것이다.

## Random Network Distillation

이 논문에서 exploration bonus $i_t$를 어떻게 정의하는지 알아보자. 먼저, 이 논문에서는 2개의 network를 사용한다.

* target - 고정 랜덤 초기화 신경망 (fixed randomly initialized network)
* predictor - observation을 예측

target network $f : \mathcal{O} \rightarrow \mathbb{R}^k$는 observation을 임베딩한다. predictor network $\hat{f} : \mathcal{O} \rightarrow \mathbb{R}^k$는 expected MSE $\lVert \hat{f}(x;\theta) - f(x) \rVert^2$를 최소화하도록 학습된다. prediction error는 predictor가 학습했던 것과 비슷하지 않은 새로운 state에 대해서 높을 것이다. 이를 통해 exploration을 도울 수 있다.

### Prediction Error

prediction error의 요인은 아래와 같다.

1. Amount of training data - 비슷한 example을 적게 관찰 했을 때
2. Stochasticity - target function이 stochastic할 때
3. Model misspecification - 반드시 필요한 정보를 놓쳤거나 target function의 복잡성에 맞추기 어려울 때
4. Learning dynamics - target function을 가장 잘 근사하는 predictor를 찾는데 실패할 때

위 첫번째 요소는 prediction error를 exploration bonus로 사용하게 하는 근본적 요인이다. 만약 예측 문제가 forward dynamics ($s_t$와 $a_t$를 통해 $s_{t+1}$을 예측하는 모델) 일 경우 두번째 요소는 'noisy-TV' 문제를 일으킨다. deterministic한 transition보다 stochastic한 transition 예측이 어려운 건 너무나 당연하다. 또한 세번째 요소 역시 부적절하다.

RND는 target network가 deterministic하게 선택되고, predictor network의 model-class 내에 있기 때문에 두번째와 세번째 요소를 피할 수 있다.

### Combining Intrinsic and Extrinsic Returns

이 논문에서는 intrinsic reward를 non-episodic return으로 좋다고 주장한다. 그 이유는 아래와 같다.

* agent의 intrinsic return은 미래에 발견할 수도 있는 모든 새로운 상태와 관련됨
* episodic intrinsic reward는 정보 누락을 발생시킴
* 이 접근법은 인간이 게임을 탐색할 때와 유사함
* episodic하다면 탐색 도중 game over 시 return이 0이 되기 때문에 risk 감수를 꺼리게 됨

그러나 extrinsic reward에 대해서는 episodic return으로 다루는 것이 좋다고 주장한다. 그 이유는 만약 게임 시작 근처에서 reward를 발견할 경우, 그 reward를 계속 획득하기 위해 의도적으로 game over를 반복적으로 당하도록 악용할 것이다.

그렇다면 어떻게 intrinsic reward $i_t$의 non-episodic stream과 extrinsic reward $e_t$의 episodic stream을 적절히 결합할 수 있을까? 이 논문에서는 extrinsic return $R_E$와 intrinsic return $R_I$ 각각을 더한 $R = R_E + R_I$를 관찰한다. 즉, 각 return에 대한 value $V_E$와 $V_I$를 구한 뒤 value function $V = V_E + V_I$로 결합한다. 이러한 아이디어로 서로 다른 discount factor를 사용한 reward stream을 결합할 수도 있다.

episodic과 non-episodic reward stream의 결합 혹은 서로 다른 discount factor를 가진 reward stream의 결합을 하지 않더라도, value function에 대한 추가적인 supervisory signal의 존재 때문에 여전히 value function을 분리하는 것에 이점이 있다. 이는 특히 exploration bonus에 중요한데 extrinsic reward function은 stationary한 반면 intrinsic reward function은 non-stationary하기 때문이다.

### Reward and Observation Normalization

prediction error를 exploration bonus로 사용할 때, 환경이 달라지거나 다른 순간에 있을 때 reward 크기가 너무 달라진다는 문제가 있다. reward를 일관된 크기로 유지하기 위해 intrinsic return의 표준편차 추정치로 나눔으로써 정규화를 수행한다.

observation도 정규화를 수행한다. observation을 정규화하지 않을 경우 임베딩의 분산이 극도로 낮아 입력에 대한 정보가 전혀 전달되지 않을 수 있다. 이를 위해 observation에 평균을 빼고 표준편차로 나눈 뒤 -5와 5 사이의 범위로 clipping한다. 정규화 파라미터 (평균, 표준편차)는 최적화 시작 전에 random agent로 약간의 step을 통해 초기화된다.