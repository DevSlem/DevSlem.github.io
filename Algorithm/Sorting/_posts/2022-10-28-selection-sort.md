---
title: "[알고리즘] 선택 정렬"
tags:
    - [Algorithm, Sorting]
last_modified_at: 2022-10-28
---

선택 정렬은 가장 간단한 컨셉을 가지는 정렬 방법 중 하나이다. 선택 정령의 특징은 아래와 같다.

* 비교 기반
* in-place
* 시간 복잡도: $O(n^2)$
* unstable

## Key Idea

오름차순으로 정렬한다고 할 때 선택 정렬의 핵심 아이디어 다음과 같다.

1. 가장 작은 원소를 선택 해 0번 원소와 교환
2. 그 다음 작은 원소를 선택 해 1번 원소와 교환
3. 위 과정을 총 $n-2$번 원소까지 총 $n - 1$번 반복

## Example

아래는 선택 정령의 과정을 보여주는 예시이다. bold체로 표시된 왼쪽 영역은 정렬이 완료되었음을 의미한다.

|List|Smallest Element|
|:---:|:---:|
|**11** 25 22 64 12|11|
|**11 12** 22 64 25|12|
|**11 12 22** 64 25|22|
|**11 12 22 25** 64|25|
|**11 12 22 25 64**|64|

위 예시에서 마지막 원소 64에 대해서는 굳이 과정을 거치지 않아도 된다.

## Algorithm

아래는 선택 정렬 알고리즘이다. $A_i$는 `A[i]`와 동일한 의미이며, $\arg\min$은 가장 작은 원소의 인덱스를 의미한다.

> ##### $\text{Algorithm: Selection sort}$  
> $$
> \begin{align*}
> & \textstyle \text{Input: an array $A$, the number of elements $n$} \\
> \\
> & \textstyle \text{Loop for $i = 0, 1, \dots, n - 2$:} \\
> & \textstyle \qquad m \leftarrow \arg\min A_{i:n} \qquad \text{($A_{l:r}$ is from $A_l$ to $A_{r-1}$)} \\
> & \textstyle \qquad \text{Swap($A_i$, $A_m$)} \\
> \end{align*}
> $$

## C++ Code

위 알고리즘을 구현한 C++는 아래와 같다. $\arg\min$은 내부 루프를 통해 구할 수 있다.

```c++
void selection_sort(int a[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        int m = i; // the index of smallest element
        for (int j = i + 1; j < n; j++)
        {
            if (a[j] < a[m])
                m = j;
        }
        swap(a[i], a[m]);
    }
}
```

## References

[1] Wikipedia. [Selection sort](https://en.wikipedia.org/wiki/Selection_sort).