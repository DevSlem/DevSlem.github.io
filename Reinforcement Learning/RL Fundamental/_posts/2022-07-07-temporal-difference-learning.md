---
title: "Temporal-Difference Learning"
tags: [RL, AI]
date: 2022-07-07
last_modified_at: 2022-07-07
sidebar:
    nav: "rl"
---

이 포스트에서는 RL에서 반드시 알아야 하는 RL의 핵심인 Temporal-Difference learning method를 소개한다.

## What is TD learning

*Temporal-Difference* (TD) learning method는 Monte Carlo (MC) method와 Dynamic Programming (DP)의 아이디어를 결합한 방법이다. TD method는 아래와 같은 특징을 가지고 있다.

* MC method처럼 model 없이 experience로부터 학습이 가능하다. (agent는 environment dynamics $p(s',r \vert s,a)$를 모른다.)
* DP처럼 다른 학습된 추정치를 기반으로 추정치를 update한다. 즉, bootstrap이다.
* Generalized Policy Iteration (GPI)를 따른다.

TD method는 DP와 MC method의 치명적 단점들을 극복한 방법이다. GPI의 *control*에서는 약간의 차이만 있을 뿐 거의 비슷하다. 핵심은 GPI의 *prediction* 부분이며 여기서 큰 차이를 보인다. TD method가 어떻게 prediction을 다루는지 알아보자.

## TD Prediction

TD와 MC method의 공통점은 prediction problem을 해결하기 위해 sample experience를 사용한다는 점이다. MC method의 가장 큰 단점은 value function을 추정하기 위해 return을 구해야 했기 때문에 episode의 종료를 기다려야 한다는 문제가 있었다. every-visit MC method의 value function을 추정하는 단순한 update rule은 아래와 같다.

$$
\begin{align}
    V(S_t) &\leftarrow V(S_t) + \alpha \Big[G_t - V(S_t) \Big] \\
    V(S_t) &\leftarrow (1 - \alpha)V(S_t) + \alpha G_t
\end{align}
$$

$G_t$는 time step $t$에 대한 return으로 MC method의 *target*이다. $\alpha$는 step-size parameter 혹은 weight이다. 위 수식을 *constant*-$\alpha$ MC라고도 부른다. 

> 참고로 이 포스트에서는 일반적인 value function은 소문자로 (e.g. $v$) 표기하며, value function의 추정치임을 명확하게 나타낼 때는 대문자로 (e.g. $V$) 표기한다.
{: .prompt-info}

위의 첫 번째 수식은 incremental한 형식으로 일반적인 형태는 아래와 같다.

$$
\textit{NewEstimate} \leftarrow \textit{OldEstimate} + \textit{StepSize} \Big[\textit{Target} - \textit{OldEstimate} \Big]
$$

TD method는 앞서 언급했듯이 DP의 bootstrap 속성을 가져왔다. TD method는 MC method처럼 episode의 종료를 기다릴 필요 없이 next time step까지만 기다리면 된다. next time step $t+1$에서 TD method는 즉시 target을 형성할 수 있다. 아래는 TD method의 간단한 update rule이다.

$$
V(S_t) \leftarrow V(S_t) + \alpha \Big[R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \Big]
$$

$R_{t+1}$은 획득한 reward, $\gamma$는 discount factor이다. TD method는 reward $R_{t+1}$과 이미 존재하는 next state value 추정치 $V(S_{t+1})$을 통해 현재 state의 value function을 즉시 업데이트 한다. 특히 업데이트에 사용되는 TD method의 target을 *TD target*이라고 하고, TD target과 현재 state의 value 추정치와의 차이를 *TD error*라고 한다. 특히 TD error는 RL에서 중요한 형식으로 RL 전반에 걸쳐 다양한 형태로 나타난다.

$$
\textit{TD target} \doteq R_{t+1} + \gamma V(S_{t+1})
$$

$$
\textit{TD error } \delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

이러한 TD method를 *TD(0)* 혹은 *one-step TD*라고 하는데 TD($\lambda$)나 $n$-step TD method의 특수 case이다. 아래는 TD(0)에 대한 알고리즘이다.

> ##### $\text{Algorithm: Tabular TD(0) for estimating } v_\pi$
> $\text{Input: the policy } \pi \text{ to be evaluated}$  
> $\text{Algorithm parameter: step size } \alpha \in (0, 1]$  
> $\text{Initialize } V(s) \text{, for all } s \in \mathcal{S}^+ \text{, arbitrarily except that } V(\textit{terminal}) = 0$  
> 
> $\text{Loop for each episode: }$  
> $\qquad \text{Initialize } S$  
> $\qquad \text{Loop for each step of episode:}$
> $\qquad\qquad A \leftarrow \text{action given by } \pi \text{ for } S$  
> $\qquad\qquad \text{Take action } A \text{, observe } R, S'$  
> $\qquad\qquad V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]$  
> $\qquad\qquad S \leftarrow S'$  
> $\qquad \text{until } S \text{ is terminal}$

아래는 TD(0)에 대한 backup diagram이다.

![](/assets/images/rl-td-backup-diagram.png){: w="43%"}
_Fig 1. TD(0) backup diagram.  
(Image source: Robotic Sea Bass. [An Intuitive Guide to Reinforcement Learning](https://roboticseabass.com/2020/08/02/an-intuitive-guide-to-reinforcement-learning/).)_  

backup diagram의 맨 위 state node에 대한 value 추정치는 next state로의 **딱 1번의 sample transition**을 통해 즉각적으로 update된다.  

## TD Control

우리는 앞서 TD prediction을 통해 value function을 추정하는 방법을 알아보았다. 이제 GPI에 따라 TD control을 통해 policy를 update할 것이다. MC method와 마찬가지로 sampling을 통해 학습하기 때문에 exploration과 exploitation에 대한 trade off 관계를 고려해야 하며, 이를 수행하는 방법으로 TD method 역시 on-policy와 off-policy method가 있다.

MC method에서 state value $v_\pi$를 추정할 경우 environment에 대한 지식인 state transition probability distribution을 알아야 policy improvement를 수행할 수 있었다.[^1] 이는 TD method에도 동일하게 적용된다. 다행이 이러한 문제는 $v_\pi$ 대신 action value $q_\pi$를 directly하게 추정하면 해결할 수 있다. $q_\pi$를 추정하는 것의 장점은 environment에 대한 지식이 필요가 없어지는 것 뿐만 아니라 [TD Prediction](#td-prediction)에서의 state value 추정과 본질적으로 같기 때문에 아래 그림과 같이 단지 state에서 state-action pair sequence로 대체하기만 하면 된다. 

![](/assets/images/rl-sutton-state-action-sequence.png)
_Fig 2. State-action pair sequence.  
(Image source: Sec 6.4 Sutton & Barto (2017).)_  

따라서 앞으로 알아볼 TD method algorithm은 모두 action value $q_\pi$를 추정한다. 이 때 TD method는 bootstrap하기 때문에 다른 학습된 next state에서의 action value 추정치를 기반으로 현재 state-action pair의 $Q(s,a)$를 추정한다. TD method algorithm들은 다른 학습된 action value 추정치를 고려하는 방식에 따라 구분된다.

## Sarsa

Sarsa는 가장 기본적인 TD on-policy method이다. 현재 state-action pair의 action value $Q(S_t,A_t)$를 추정할 때 다른 학습된 next state-action pair에 대한 $Q(S_{t+1},A_{t+1})$을 현재 policy $\pi$에 따라 선택한다. 이에 대한 update rule은 아래와 같다.

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \Big]
$$

당연하지만 $S_{t+1}$이 terminal state일 경우 $Q(S_{t+1}, A_{t+1})$은 0이다. 위 update rule을 [TD Prediction](#td-prediction)에서 보았던 TD method의 state value에 대한 update rule과 비교해볼 때 단지 state value 추정치 $V(S)$를 action value $Q(S, A)$로 대체했을 뿐임을 확인할 수 있다. $Q(S_{t+1}, A_{t+1})$을 구할 때 next state $S_{t+1}$에서 현재 policy $\pi$에 따라 action $A_{t+1}$을 선택하고 그 action의 value를 선택함을 꼭 기억해야 한다. 아래는 위 update rule의 backup diagram이다.

![](/assets/images/rl-sutton-sarsa-backup-diagram.png)
_Fig 3. Sarsa backup diagram.  
(Image source: Sec 6.4 Sutton & Barto (2017).)_  

모든 on-policy method에서는 experience 생성에 사용된 behavior policy $\pi$에 대한 $q_\pi$를 추정함과 동시에, 추정된 $q_\pi$에 관해 behavior policy $\pi$를 greedy한 방향으로 update한다. Sarsa가 수렴하기 위해서는 exploration이 잘 수행되어야 하기 때문에 주로 $\epsilon$-soft policy류의 방법을 사용한다. 아래는 Sarsa algorithm이다.

> ##### $\text{Sarsa (on-policy TD control) for estimating } Q \approx q_\ast$  
> $\text{Algorithm parameters: step size }  \alpha \in (0,1], \text{ small } \epsilon > 0$  
> $\text{Initialize } Q(s,a) \text{, for all } s \in \mathcal{S}^+, a \in \mathcal{A}(s) \text{, arbitrarily except that } Q(\textit{terminal},\cdot) = 0$  
> 
> $\text{Loop for each episode:}$  
> $\qquad \text{Initialize } S$  
> $\qquad \text{Choose } A \text{ from } S \text{ using policy derived from } Q \text{ (e.g., } \epsilon \text{-greedy)}$  
> $\qquad \text{Loop for each step of episode:}$  
> $\qquad\qquad \text{Take action } A \text{, observe } R, S'$  
> $\qquad\qquad \text{Choose } A' \text{ from } S' \text{ using policy derive from } Q \text{ (e.g., } \epsilon \text{-greedy)}$  
> $\qquad\qquad Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma Q(S',A') - Q(S,A)]$  
> $\qquad\qquad S \leftarrow S'; \ A \leftarrow A';$  
> $\qquad \text{until } S \text{ is terminal}$

## Footnotes

[^1]: [Monte Carlo Estimation of Action Values](../monte-carlo-methods/#monte-carlo-estimation-of-action-values)