---
title: "게임 물리 요약 (Game Physics Summary)"
excerpt: "게임에 적용되는 기본적인 물리 법칙에 대해 소개한다."
categories: [Game Development, Math Physics]
date: 2022-05-22
last_modified_at: 2022-05-22
math: true
---

이 포스트에서는 게임에 적용되는 기본적인 물리 법칙에 대해 소개한다.

## 등가속도 운동

속도는 시간에 따른 위치의 변화율, 가속도는 시간에 따른 속도의 변화율이다. 시간 $t$에 대한 위치를 $\textbf{x}(t)$, 속도를 $\textbf{v}(t)$, 가속도를 $\textbf{a}(t)$라고 할 때 아래와 같은 관계가 성립한다.

$$
\textbf{a}(t) = \dot{\textbf{v}}(t) = \ddot{\textbf{x}}(t)
$$

$\dot{f}$는 함수 $f$에 대한 미분이다. 등가속도 운동의 속도는 아래와 같이 정의된다.

$$
\textbf{v}(t) = \textbf{v}_0 + \int_0^t \textbf{a}(t)dt = \textbf{v}_0 + \textbf{a}_0t
$$

$\textbf{v}_0$는 $t=0$에서의 속도이다. 위 속도를 바탕으로 위치를 결정할 수 있다.

$$
\textbf{x}(t) = \textbf{x}_0 + \int_0^t \textbf{v}(t)dt = \textbf{x}_0 + \int_0^t(\textbf{v}_0 + \textbf{a}_0t) = \textbf{v}_0t + \dfrac{1}{2}\textbf{a}_0t^2
$$

$\textbf{x}_0$는 $t=0$에서의 위치이다.

## 포물선 운동

포물선 운동은 중력에 의해 영향을 받는 물체의 운동으로 3차원 공간에서 중력이 작용하는 axis를 $z$라고 가정할 때 중력 가속도 $\textbf{g} = [0,0,-g]$로 정의된다. 여기서 $g$는 중력 상수로 지구의 지표면을 기준으로 약 $9.8 \ m/s^2$이며 아래쪽 방향으로 작용한다. 포물선 운동에서 위치 $\textbf{x}(t)$는 아래와 같이 정의된다.

$$
\textbf{x}(t) = \textbf{x}_0 + \textbf{v}_0t + \dfrac{1}{2}\textbf{g}t^2
$$

위치 $\textbf{x} = [x, y, z]$라고 할 때 위 수식의 각 성분은 아래와 같이 구성된다.

$$
\textbf{x}(t) =
\begin{bmatrix}
  x(t) \\
  y(t) \\
  z(t)
\end{bmatrix} =
\begin{bmatrix}
  x_0 + v_xt \\
  y_0 + v_yt \\
  z_0 + v_zt - \dfrac{1}{2}gt^2
\end{bmatrix}
$$

중력에 의한 운동에서 발사체가 최대 높이에 도달할 때의 순간 속도 $v = 0$이다. 최대 높이에 도달하는 시간을 $t$라고 할 때 아래와 같은 방식으로 구할 수 있다.

$$
\dfrac{d}{dt}z(t) = v_z - gt = 0 \quad \therefore t = \dfrac{v_z}{g}
$$

위에서 구한 $t$값을 바탕으로 최대 높이 $h$를 구할 수 있다.

$$
h = z_0 + v_z\dfrac{v_z}{g} - \dfrac{1}{2}g\Big(\dfrac{v_z}{g}\Big)^2 = z_0 + \dfrac{v_z^2}{2g}
$$

발사체가 발사된 이후 원래 높이 $z_0$로 내려왔을 때의 시간 $t \neq 0$는 아래와 같이 구할 수 있다.

$$
z(t) = z_0 + v_zt - \dfrac{1}{2}gt^2 = z_0 \quad \therefore t = \dfrac{2v_z}{g}
$$

이를 바탕으로 발사체가 원래 높이 $z_0$로 내려올 때까지의 이동한 수평 거리 $r$은 아래와 같이 구할 수 있다.

$$
r_x = v_xt = \dfrac{2v_xv_z}{g} \\
r_y = v_yt = \dfrac{2v_yv_z}{g}
$$

초기 속력 $s$, 최대 높이 $h$가 주어졌을 때 발사체의 발사각 $\theta$는 아래와 같다.

$$
h = z_9 + \dfrac{v_z^2}{2g} = z_0 + \dfrac{(s \sin \theta)^2}{2g} \quad \therefore \theta = \sin^{-1} \Big(\dfrac{1}{s}\sqrt{2g(h - z_0)} \Big)
$$

초기 속력 $s$, 속도 $\textbf{v}$의 $xy$ 평면에 정사영된 벡터의 크기를 $v_{xy}$, 원래 높이로 내려올 때까지의 $xy$평면에서의 수평 이동 거리 $r_{xy} = \lVert r_x + r_y \rVert_2$이 주어져 있을 때 발사체의 발사각 $\theta$는 아래와 같다.

$$
r_{xy} = \dfrac{2v_{xy}v_z}{g} = \dfrac{2(s \cos \theta)(s \sin \theta)}{g} = \dfrac{s^2 \sin 2\theta}{g} \quad \therefore \theta = \dfrac{1}{2}\sin^{-1}\dfrac{r_{xy}g}{s^2}
$$

## Force

물체에 힘이 가해지면 운동량과 가속도가 변한다. 여러 힘이 가해질 경우 하나의 전체 힘으로 합할 수 있다.

$$
F_{\text{total}} = \sum F_i
$$

뉴턴 법칙은 아래와 같다.

* 제 1법칙: 관성의 법칙
* 제 2법칙: 가속도 법칙 - $F = ma$
* 제 3법칙: 반작용 법칙 - $F_{ij} = -F_{ji}$

### 중력

중력은 지구 표면에서 작용하는 힘이다. 이때 중력가속도 $g = -9.8 \ m/s^2$이다.

$$
F = mg
$$

만유인력은 두 물체 사이에 작용하는 힘으로 두 물체 사이의 거리를 $r$, 중력상수 $G = 6.673 \times 10^{-11}$, 두 물체의 질량을 각각 $m$, $M$이라고 할 때 그 수식은 아래와 같다.

$$
F = G\dfrac{mM}{r^2}
$$

### 수직항력

수직항력은 표면이 물체에 가하는 반작용 힘이다. 항상 표면에 수직인 방향으로 작용한다. 중력 가속도가 $g$, 경사각이 $\theta$일 때 수직항력 $F_n$은 일반적으로 아래와 같다.

$$
F_n = -mg\cos\theta
$$

<div style="text-align: center">
<img width="30%" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Friction_relative_to_normal_force_%28diagram%29.png/330px-Friction_relative_to_normal_force_%28diagram%29.png" style="background-color:white">
<figcaption>Fig 1. Normal force<br>
(Image source: Wikipedia <a href="https://en.wikipedia.org/wiki/Normal_force">Normal force</a>)</figcaption>
</div>

### 마찰력

마찰력은 표면 상의 물체의 운동을 방해하는 힘으로 물체에 가해지는 접선 방향 힘의 반대 방향으로 작용한다. 마찰력의 크기는 표면과 특성에 의존한다. 물체의 질량에 비례하나 접촉 면적과는 무관하다. 마찰력 $F_f$는 수직항력 $F_n$으로부터 계산 가능하다.

$$
F_f = \mu F_n
$$

$\mu$는 마찰계수로 표면의 특성에 따라서 정해진다. 마찰력은 운동마찰려과 정지마찰력이 있는데 둘 중 하나만 적용된다. 운동마찰은 서로 상대적으로 움직이는 두 표면 사이에서 발생하는 힘으로 각자의 운동에 대한 저항으로 작용한다. 운동마찰력을 $F_k$, 운동마찰계수를 $\mu_k$라고 할 때 수식은 아래와 같다.

$$
F_k = \mu_k F_n
$$

정지마찰은 표면이 정지된 물체를 움직이지 못하도록 붙들고 있는 힘이다. 정지마찰력을 $F_s$, 정지마찰계수를 $\mu_s$라고 할 때 수식은 아래와 같다.

$$
F_s = \mu_s F_n
$$

정지마찰력 $F_s$일 때 물체에 가해지는 힘 $F$의 크기가 $F_s$보다 큰 순간 물체가 움직이기 시작한다. 그때부터 정지마찰력 $F_s$는 사라지고 운동마찰력 $F_k$가 작용한다. 대체로 $F_k < F_s$인데 정지된 물체를 움직이게 하는 것이 이미 움직이는 물체를 계속 움직이게 하는 것보다 대체로 더 힘들다.

### 스프링 운동

스프링 운동은 비강체 물체를 모델링하기 위해 주로 쓰인다. rigid body의 움직임을 제약하는 joint를 생성해 처리한다. 스프링 힘(Hooke's law) $F_s$은 아래와 같다.

$$
F_s = -k_sx
$$

<!-- <div style="text-align: center">
<img width="40%" src="/assets/images/spring-system.png">
<figcaption>Fig 2. 스프링 시스템<br>
(Image source: INU - Game Programming Lecture)</figcaption>
</div> -->

$k_s$는 강성계수이며 두 점의 위치를 각각 $p_0$, $p_1$이라고 할 때 두 물체의 변위 $x = p_1 - p_0$이다.  두 물체의 원래 거리를 $d$라고 할 때 스프링 힘은 아래와 같다.

$$
F_s = -k_s(\lVert x \rVert_2 - d)\dfrac{x}{\lVert x \rVert_2}
$$

두 지점의 거리 $\lVert x \rVert_2$가 $d$보다 가까우면 밀어내고, $d$보다 멀면 당기도록 작용한다.

감쇠력은 에너지 손실을 도입하여 스프링이 무한히 oscillate하지 않도록 하게 한다. 감쇠력 $F_d$는 아래와 같다.

$$
F_d = -k_dv
$$

$k_d$는 점성 감쇠계수이다. 각 위치에서의 속도 벡터를 $v_0$, $v_1$이라고 할 때 속도 변화 $v = v_1 - v_0$이다. 최종적으로 감쇠된 스프링 시스템은 아래와 같이 표현할 수 있다.

$$
F = F_s + F_d = -k_sx -k_dv
$$

### 운동량

운동량 $p$는 질량과 속도를 곱한 벡터이다.

$$
p = mv
$$

두 물체가 충돌 시 운동량은 보존 되며 이를 운동량 보존 법칙이라고 한다.

$$
m_1v_1 + m_2v_2 = m_1v_1' + m_2v_2'
$$

$v_1$, $v_2$는 충돌 전 각 물체의 속도이며 $v_1'$, $v_2'$은 충돌 후 각 물체의 속도이다.

## 충돌

충돌을 검사하는 일반적인 방법은 다음과 같이 크게 3가지가 있다.

* Particle vs Plane
* Plane vs Plane
* Edge vs Plane

이 중 Particle vs Plane 충돌이 구현하기 가장 쉬운편이다. 따라서 이 포스트에서는 Particle vs Plane 충돌에 대해 알아본다.

### Particle vs Plane Collision Detection

3차원 공간에서 다음과 같은 요소들이 존재한다.

* $\text{x}$ - particle의 위치
* $\text{p}$ - plane 위의 임의의 점
* $\text{n}$ - 평면의 정방향 normal vector  

<!-- <div style="text-align: center">
<img width="40%" src="/assets/images/collision-detection-particle-plane.png">
<figcaption>Fig 3. particle vs plane collision detection<br>
(Image source: INU - Game Programming Lecture)</figcaption>
</div> -->

이때 $(\text{x} - \text{p}) \cdot \text{n}$가 0보다 크면 particle은 plane과 $\text{x}$는 충돌 전 상태, 0이면 plane과 접촉, 0보다 작으면 평면을 통과한 상태이다. 이는 내적 자체가 각도 정보를 포함하고 있기 때문이다.  

### Particle vs Plane Collision Response

실제 collision detection 시  오차로 인해 내적값이 0이 되는 경우를 찾기 어렵다. 따라서 내적 값이 양수 -> 음수로 변할 떄 충돌했다고 가정한다. 즉, 평면을 관통했을 때 충돌했다고 판정한다. 그 후 관통한 particle을 평면 위로 이동시킨 후 충돌에 대한 후처리로 particle이 튕겨져나가는 물리적 처리를 한다.

$\text{v}$ - current velocity of the particle
$\text{n}$ - normal vector on the illegal side of the plane (direction vector whose magnitude is 1)

이때 속도 $\text{v}$의 normal 성분인 $\text{v}_n$은 아래와 같다.

$$
\text{v}_n = (\text{v} \cdot \text{n})\text{n}
$$

속도 $\text{v}$의 tangential 성분도 유도 가능하다.

$$
\text{v}_t = \text{v} - \text{v}_n
$$

위 두 벡터를 바탕으로 particle이 튕겨져나가는 bounced response $\text{v}_b$를 구할 수 있다.

$$
\begin{aligned}
  \text{v}_b &= \text{v}_t - \text{v}_n \\
  &= (1 - k_f)\text{v}_t - k_r\text{v}_n
\end{aligned}
$$

$k_f$는 마찰계수, $k_r$은 복원계수이다.

## 물리상태 계산

물체의 이전 물리 상태(위치, 속도 등)와 가해진 힘을 알면 **시간에 대한 적분을 통해 물체의 이후 상태를 결정**할 수 있다. 아래는 움직임에 대한 수식이다.

$$
F = ma \rightarrow a = \dfrac{F}{m} \\
\dfrac{dv}{dt} = a \rightarrow dv = a \ dt \\
\dfrac{dx}{dt} = v \rightarrow dx = v \ dt
$$

### Runge-Kutta 적분법

4차 Runge-Kutta 적분법 (RK4)을 활용하면 상당히 정확한 수치해석적 적분을 구현할 수 있다. 아래와 같은 function $f$와 초기값 $t_0$, $y_0$가 있다고 하자.

$$
\dfrac{dy}{dt} = f(t, y), \quad y(t_0) = y_0
$$

$y$는 time $t$에 대한 알려지지 않은 함수로 근사화할 대상이다. step-size $h > 0$일 떄 아래와 같은 방식으로 근사화할 수 있다.

$$
\begin{aligned}
  y_{n+1} &= y_n + \dfrac{1}{6}h(k_1 + 2k_2 + 2k_3 + k_4), \\
  t_{n+1} &= t_n + h
\end{aligned}
$$

$y_{n+1}$은 $y(t_{n+1})$에 대한 RK4 근사값으로 next value $y_{n+1}$은 current value $y_n$과 4개의 기울기 요소의 weighted average와의 합으로 구성된다. 기울기 요소는 아래와 같다.

* $k_1 = f(t_n, y_n)$
* $k_2 = f \Big(t_n + \dfrac{h}{2}, y_n + h\dfrac{k_1}{2} \Big)$
* $k_3 = f \Big(t_n + \dfrac{h}{2}, y_n + h\dfrac{k_2}{2} \Big)$
* $k_4 = f(t_n + h, y_n + hk_3)$

아래는 RK4 적분법에서의 기울기 요소를 나타낸다.

<div style="text-align: center">
<img width="40%" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Runge-Kutta_slopes.svg/900px-Runge-Kutta_slopes.svg.png" style="background-color:white">
<figcaption>Fig 2. Runge-Kutta slopes<br>
(Image source: Wikipedia <a href="https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods">Runge-Kutta methods</a>)</figcaption>
</div>

<!-- 위 내용을 바탕으로 실제 코드로 구현해보자.

```c++
struct State {
  float x; // position
  float v; // velocity
}

struct Derivative {
  float dx; // derivative of position: velocity
  float dv; // derivative of velocity: acceleration
}

Derivative Evaluate(const State& initial, float t, float dt, const Derivative& d) {
  State state;
  state.x = initial.x + d.dx * dt;
  state.v = initial.v + d.dv * dt;
}
``` -->

## References

[1] Incheon National University - Game Programming Lecture  
[2] Wikipedia [Normal force](https://en.wikipedia.org/wiki/Normal_force)  
[3] Wikipedia [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
