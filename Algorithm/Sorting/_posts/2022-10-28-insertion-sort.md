---
title: "[알고리즘] 삽입 정렬"
tags:
    - [Algorithm, Sorting]
last_modified_at: 2022-10-28
---

삽입 정렬은 선택된 원소를 이미 정렬된 영역에 삽입하는 방식의 간단한 정렬 알고리즘으로, 실제 사람이 카드 게임 시 카드를 정렬할 때와 유사한 방식이다. 특징은 아래와 같다.

* 비교 기반
* in-place
* 시간 복잡도: $O(n^2)$
* stable

## Key Idea

삽입 정렬의 과정은 아래와 같다.

1. $i$번째 원소를 선택
2. $0, 1, \dots, i-1$번째 원소는 이미 오름차순으로 정렬되어 있음
3. $i-1$부터 $0$번째까지의 $j$번째 원소를 선택된 원소와 비교 후 선택 된 원소가 작을 경우 $j$번째 원소를 오른쪽으로 이동
4. 3번 과정을 선택된 원소가 $j$번째 원소보다 작을 동안 반복
5. $j + 1$번째에 선택된 원소를 삽입
6. 위 과정을 모든 원소에 대해 반복

## Example

아래 움짤을 보면 삽입 정렬이 동작하는 방식을 쉽게 이해할 수 있다.

![](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)
_Fig 1. A graphical example of insertion sort.  
(Image source: Wikipedia. [Insertion sort](https://en.wikipedia.org/wiki/Insertion_sort#/media/File:Insertion-sort-example-300px.gif).)_  

## Algorithm

[Key Idea](#key-idea)와 위 예시를 바탕으로 아래와 같이 알고리즘을 작성할 수 있다.

> ##### $\text{Algorithm: Insertion sort}$  
> $$
> \begin{align*}
> & \textstyle \text{Input: an array $A \in \mathbb{R}^n$, the number of elements $n$} \\
> \\
> & \textstyle \text{Loop for $i=1,2,\dots,n-1$:} \\
> & \textstyle \qquad x \leftarrow A_i \\
> & \textstyle \qquad j \leftarrow i - 1 \\
> & \textstyle \qquad \text{Loop while $j \geq 0$ and $A_j > x$:} \\
> & \textstyle \qquad\qquad A_{j+1} \leftarrow A_j \\
> & \textstyle \qquad\qquad j \leftarrow j - 1 \\
> & \textstyle \qquad A_{j+1} \leftarrow x \\
> \end{align*}
> $$

## C++ Code

아래는 위 알고리즘을 C++로 작성한 코드이다.

```c++
void insertion_sort(int a[], int n)
{
    for (int i = 1; i < n; i++)
    {
        int x = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > x)
        {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = x;
    }
}
```

## References

[1] Wikipedia. [Insertion sort](https://en.wikipedia.org/wiki/Insertion_sort).
