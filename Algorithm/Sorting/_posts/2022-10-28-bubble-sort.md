---
title: "[알고리즘] 버블 정렬"
tags:
    - [Algorithm, Sorting]
last_modified_at: 2022-10-28
---

버블 정렬은 인접한 두 원소를 비교하여 정렬하는 간단한 방식의 알고리즘이다. 버블 정렬의 특징은 아래와 같다.

* 비교 기반
* in-place
* 시간 복잡도: $O(n^2)$
* stable

## Key Idea

컨셉은 간단하다. 오름차순으로 정렬한다고 할 떄, $i$번째 원소와 $i+1$번째 원소를 비교해 $i$번째 원소가 더 크면 교환한다. $i$를 계속 키워나가다 보면 가장 큰 원소가 맨 마지막 위치에 있게 된다.

## Example

배열 [ 5 1 4 2 8 ]이 있을 때 정렬은 아래와 같이 수행된다.

#### First Pass

$i=4$일 때:

[ **5 1** 4 2 8 ] $\rightarrow$ [ **1 5** 4 2 8 ]  
[ 1 **5 4** 2 8 ] $\rightarrow$ [ 1 **4 5** 2 8 ]  
[ 1 4 **5 2** 8 ] $\rightarrow$ [ 1 4 **2 5** 8 ]  
[ 1 4 2 **5 8** ] $\rightarrow$ [ 1 4 2 **5 8** ]

#### Second Pass

$i=3$일 때:

[ **1 4** 2 5 8 ] $\rightarrow$ [ **1 4** 2 5 8 ]  
[ 1 **4 2** 5 8 ] $\rightarrow$ [ 1 **2 4** 5 8 ]  
[ 1 2 **4 5** 8 ] $\rightarrow$ [ 1 2 **4 5** 8 ]  

위 과정을 $i=1$일 때까지 반복하면 된다.


## Algorithm

위 예시를 바탕으로 알고리즘을 작성해보자. $A_i$는 `A[i]`와 동일한 의미이다.

> ##### $\text{Algorithm: Bubble sort}$  
> $$
> \begin{align*}
> & \textstyle \text{Input: an array $A$, the number of elements $n$} \\
> \\
> & \textstyle \text{Loop for $i = n - 1, n - 2, \dots, 1$:} \\
> & \textstyle \qquad \text{Loop for $j = 0, 1, \dots, i - 1$:} \\
> & \textstyle \qquad\qquad \text{If $A_j > A_{j+1}$, then:} \\
> & \textstyle \qquad\qquad\qquad \text{Swap($A_j$, $A_{j+1}$)} \\
> \end{align*}
> $$

## C++ Code

위 알고리즘을 바탕으로 C++ 코드를 작성하면 아래와 같다. 위 알고리즘과 거의 동일하다.

```c++
void bubble_sort(int a[], int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        for (int j = 0; j < i; j++)
        {
            if (a[j] > a[j + 1])
                swap(a[j], a[j + 1]);
        }
    }
}
```

## References

[1] Wikipedia. [Bubble sort](https://en.wikipedia.org/wiki/Bubble_sort).
