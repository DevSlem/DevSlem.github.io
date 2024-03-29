---
title: "[알고리즘] 쉘 정렬"
tags:
    - [Algorithm, Sorting]
last_modified_at: 2022-10-31
---

쉘 정렬은 삽입 정렬을 최적화한 알고리즘으로 $h$ 간격으로 부분적으로 정렬한다. 특징은 아래와 같다.

* 비교 기반
* in-place
* 시간 복잡도: 간격 $h$에 따라 다름
* unstable

## Key Idea

기존 삽입 정렬에서 왼쪽은 정렬된 리스트, 오른쪽은 정렬이 안된 리스트로 구분했었다. 따라서 삽입 정렬은 어느 정도 정렬이 되어 있는 리스트에 대해 강력한 성능을 낼 수 있다. 이 특징을 확장한 것이 쉘 정렬이다. 

쉘 정렬은 $h$ 간격으로 원소들을 삽입 정렬을 사용해 정렬한다. $h$는 매우 큰 수에서 점점 작아지다가 최종적으로는 1로 끝난다. 중요한 점은 반드시 $h$는 1로 끝나야 한다. $h$가 작아질 수록 부분적으로 정렬된 원소들이 많아지기 때문에 삽입 정렬의 성능은 향상되는 것이 핵심이다.

$h$ 값은 알고리즘마다 다르며, $h$ 값에 따라 시간 복잡도가 달라진다. 여기서는 아래와 같이 비교적 간단한 간격을 사용할 것이다.

$$
h = \dfrac{3^k - 1}{2} < n = 1, 4, 13, 40, 121, \dots
$$

$h$는 간격, $n$은 원소 개수이다. 위 간격의 시간 복잡도는 $O(n^\frac{3}{2})$이다.

## Example

|Index|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Element|3|14|12|4|10|13|15|5|2|7|9|6|8|11|1|

먼저, 위 배열 $\mathbf{a}$에서 간격 $h=13$부터 시작한다. $(\mathbf{a}_ 0, \mathbf{a}_ {13})$에 대해 $(3, 11) \rightarrow (3, 11)$, $(\mathbf{a}_ 1, \mathbf{a}_{14})$에 대해 $(14, 1) \rightarrow (1, 14)$로 삽입 정렬을 수행한다. 

|Index|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|After 13-sorting|3|1|12|4|10|13|15|5|2|7|9|6|8|11|14|

그 다음 $h=4$이다. 마찬가지로 $(\mathbf{a}_ 0, \mathbf{a}_ 4, \mathbf{a}_ 8, \mathbf{a}_ {12})$, $(\mathbf{a}_ 1, \mathbf{a}_ 5, \mathbf{a}_ 9, \mathbf{a}_ {13})$, $(\mathbf{a}_ 2, \mathbf{a}_ 6, \mathbf{a}_ {10}, \mathbf{a}_ {14})$, $(\mathbf{a}_ 3, \mathbf{a}_ 7, \mathbf{a}_ {11})$에 대해 삽입 정렬을 수행한다. 그 결과는 아래와 같다.

|Index|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|After 4-sorting|2|1|9|4|3|7|12|5|8|11|14|6|10|13|15|

마지막으로 $h=1$에 대해 삽입 정렬을 수행한다. 간격이 1이라는 것은 곧 모든 원소에 대해 한번에 삽입 정렬을 수행한다는 의미로 일반적인 삽입 정렬과 동일하다.

|Index|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|After 1-sorting|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|

## Algorithm

위 예제를 바탕으로 알고리즘을 작성해보자. 크게 2가지를 고려해야한다.

1. 시작 간격 $h$를 어떻게 설정할 것인가?
2. 각 간격에 대한 부분 리스트를 어떻게 처리할 것인가?

시작 간격 $h$는 $h$를 계속 키우다가 $h \geq n$이 되는 순간을 포착하면 된다. 두 번째가 조금 난해할 수 있는데 부분 리스트 끼리 나누어 생각하다 보면 난관에 봉착할 수 있다. 특히 위 [Example](#example)의 $h=4$일 때 어느 부분 리스트의 원소는 4개이고, 어느 부분 리스트의 원소는 3개이다. 이를 처리하기에는 상당히 까다롭다. 따라서 생각의 전환이 필요하다.

쉘 정렬은 기본적으로 삽입 정렬 기반이다. 삽입 정렬은 왼쪽은 정렬된 리스트, 오른쪽은 정렬이 안된 리스트로 구분한다. 결국 왼쪽에서 오른쪽으로 진행한다. 따라서 부분 리스트끼리 처리하는 것이 아니라, 한 칸씩 오른쪽으로 이동하면서 삽입 정렬을 수행하되 그 삽입 정렬의 간격이 $h$라고 생각한다. 구체적으로 삽입 할 원소의 인덱스를 $i$라고 할때, $i-h, i-2h, \dots$는 이미 정렬된 리스트이다. 따라서 $A_i$를 왼쪽 $h$간격의 정렬된 리스트에서 적당한 위치를 찾아 삽입한다. 이를 $i+1, i+2, \dots$에 대해 반복한다.

자 이제 알고리즘을 작성해보자. $\mathbf{a}_i$는 `a[i]`와 동일한 의미이다.

> ##### $\text{Algorithm: Shell sort}$  
> $$
> \begin{align*}
> & \textstyle \text{Input: an array $\mathbf{a} \in \mathbb{R}^n$, the number of elements $n$} \\
> \\
> & \textstyle h \leftarrow 1 \\
> & \textstyle \text{Loop while $h < n$:} \\
> & \textstyle \qquad h \leftarrow 3h + 1 \\
> \\
> & \textstyle h \leftarrow (h - 1) \div 3 \\
> & \textstyle \text{Loop while $h \geq 1$:} \\
> & \textstyle \qquad \text{Loop for $i = h, h+1, \dots$:} \\
> & \textstyle \qquad|\qquad x \leftarrow \mathbf{a}_i \\
> & \textstyle \qquad|\qquad j \leftarrow i - h \\
> & \textstyle \qquad|\qquad \text{Loop while $j \geq 0$ and $\mathbf{a}_j > x$:} \\
> & \textstyle \qquad|\qquad\qquad \mathbf{a}_{j+h} \leftarrow \mathbf{a}_j \\
> & \textstyle \qquad|\qquad\qquad j \leftarrow j - h \\
> & \textstyle \qquad|\qquad \mathbf{a}_{j+h} \leftarrow x \\
> & \textstyle \qquad \text{until $i=n-1$} \\
> & \textstyle \qquad h \leftarrow (h - 1) \div 3 \\
> \end{align*}
> $$

위 알고리즘에서 내부 루프는 간격이 $h$인 부분 리스트들에 대한 삽입 정렬이다. $h=1$로 바꿔보면 삽입 정렬 알고리즘과 완벽하게 동일함을 알 수 있다.

## C++ Code

위 알고리즘을 바탕으로 C++ 코드를 작성해보자. `dtype`은 임의의 비교 가능한 데이터 타입이다.

```c++
void shell_sort(dtype a[], int n)
{
    int h = 1;
    while (h < n)
    {
        h = 3 * h + 1;
    }

    for (h = (h - 1) / 3; h >= 1; h = (h - 1) / 3)
    {
        for (int i = h; i < n; i++)
        {
            dtype x = a[i];
            int j = i - h;
            while (j >= 0 && a[j] > x)
            {
                a[j + h] = a[j];
                j -= h;
            }
            a[j + h] = x;
        }
    }
}
```

## References

[1] Wikipedia. [Shellsort](https://en.wikipedia.org/wiki/Shellsort).