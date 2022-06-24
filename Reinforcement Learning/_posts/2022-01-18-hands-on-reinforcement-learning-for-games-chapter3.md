---
title: "Hands-On Reinforcement Learning for Games - Chapter 3"
excerpt: "Chapter 3: Monte Carlo Methods"
categories:
    - Reinforcement Learning
tags:
    - [RL, AI]
date: 2022-01-18
last_modified_at: 2022-01-18
---

# 1. 개요

**Hands-On Reinforcement Learning for Games** 강화 학습 책에 대해 공부한 내용을 정리하기 위한 포스트이기 때문에 자세히 기술하지 않을 것이다.  
참고로 이 책은 개념 위주가 아닌 실습 위주의 책이기 떄문에 개념을 자세히 설명해주지 않는다.



# 2. Model-based vs Model-free learning

* Model based: MDP에 대한 정보를 미리 아는 경우
* Model-free: MDP에 대한 정보를 모르는 경우




# 3. Monte Carlo method

계산하기 어려운 값을 수많은 확률 시행을 거쳐 추산하는 기법

#### $\pi$ 추정하기

$2 \times 2$ 크기의 사각형 안에 반지름이 1인 원이 있다고 가정해보자. 
이 사각형 영역 안에 점을 랜덤하게 찍을 것이다. 
그 점들은 원 안에 찍힐 수도, 원 밖에 찍힐 수도 있다. 
이러한 랜덤성을 이용하는 것이 ***Monte Carolo method*** 이다.  
아래 그림을 보자.

![pi 추정](/assets/images/monte-carlo-method-estimate-pi.png)

위 그림에서 초록색 점은 원 안에 찍힌 점, 빨간색 점은 원 밖에 찍힌 점이다. 사각형과 원의 넓이를 알고 있다면 우리는 다음과 같은 비례 관계를 적용해 $\pi$ 값을 추정할 수 있다.

> $4 : \pi = total : ins$  
> $\pi = \displaystyle\frac{4 \times ins}{total}$

* $ins$: 원 안에 찍힌 점의 개수
* $total$: 총 찍은 점의 개수 

#### 특징

1. 몬테 카를로 추정은 불편 추정치이다.
      * 불편 추정치: 추정치의 평균이 편향되지 않아서 수많은 시뮬레이션 횟수를 거치면 실제 참값과 같아짐을 의미
2. TD 기법보다 분산이 크다.  




# 4. RL에 적용

model이 없는 상태. 시행착오를 통해 환경을 탐험하는 알고리즘을 개발해야함. 이를 위해 Monte Carlo method를 활용.

> Essentially, our algorithm becomes an **explorer** rather than a **planner** and this is why we
now refer to it as an agent.

#### Episode

*start*에서 *termination*까지의 완전한 움직임 과정.  
하나의 에피소드가 끝난 후에 학습 및 개선이 이루어지는 방식을 ***episodic learning***이라고 함.
 
#### Monte Carlo control

평균을 샘플링 하는 방법의 차이

* First-Visit Monte Carlo: 방문한 상태에 대해 첫번째 value만 사용
* Every-Visit Monte Carlo: 방문한 상태에 대해 모든 value를 사용

Monte Carlo method는 랜덤성을 결정하기 위한 다양한 샘플링 분배를 사용한다. 이 책의 코드에서는 ***uniform*** 방식을 택했지만, real-world 환경에서는 대부분 ***Gaussian*** 샘플링을 택한다.




# 5. Prediction and control

#### Incremental means

> $\mu_k = \mu_{k-1} + \displaystyle\frac{1}{k}(x_k - \mu_{k-1})$

#### 위 식의 요소

* $\mu_k$: $k$번째까지의 평균
* $x_k$: $k$번째의 데이터

#### Greedy policy

> $\pi(a\|s) = \begin{cases} 1 \quad if \; a = a^* \\\ 0 \quad if \; a \neq a^* \end{cases}$

#### Epsilon-Greed policy

$\epsilon$이 증가할 수록 랜덤성이 부여됨. 0일 경우에는 greedy policy임.

> $\pi(a\|s) = \begin{aligned} \begin{cases} 1 - \epsilon + \displaystyle\frac{\epsilon}{\| A(s) \|} \quad &if \; a = a^* \\\ \displaystyle\frac{\epsilon}{\| A(s) \|} \quad &if \; a \neq a^* \end{cases} \end{aligned}$

#### 위 식의 요소

* $a^*$: best action
* $A(s)$: 가능한 action 개수
* $\epsilon$: 탐색을 선택할 확률(선택에 임의성을 부여)