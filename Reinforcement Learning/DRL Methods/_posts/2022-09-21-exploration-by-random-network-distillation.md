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