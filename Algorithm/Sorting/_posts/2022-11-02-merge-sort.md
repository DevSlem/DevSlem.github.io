---
title: "[알고리즘] 합병 정렬"
tags:
    - [Algorithm, Sorting]
last_modified_at: 2022-11-02
---

합병 정렬은 효율적이고 일반적인 목적으로 사용되는 [divide-and-conquer](https://en.wikipedia.org/wiki/Divide-and-conquer_algorithm) 기반의 정렬 알고리즘이다. 합병 정렬의 특징은 아래와 같다.

* 비교 기반
* non-in-place
* 시간 복잡도: $O(n \log n)$
* stable

## Key Idea

합병 정렬의 핵심 아이디어는 아래와 같다.

1. 정렬되지 않은 $n$개의 서브 리스트로 분할한다. 각 서브 리스트는 1개의 원소를 가진다.
2. 서브 리스트를 합쳐 새로운 정렬된 서브 리스트를 만든다. 이 과정은 하나의 서브 리스트가 될 때까지 반복한다.

위와 같이 비교적 간단한 아이디어임을 알 수 있다. 합병 정렬은 top-down과 bottom-up 방식 모두 존재하는데 여기서는 top-down 방식에 대해 알아볼 것이다.

먼저, 어떤 입력 리스트가 있다고 할 때 이를 재귀적으로 절반씩 서브 리스트로 분할한다. 서브 리스트의 원소가 1개가 될 떄까지 반복한다. 이제 이 서브 리스트를 정렬된 상태로 합치기만 하면 끝이다.

## Example

아래 그림은 top-down 방식의 합병 정렬 예시이다. 빨간색은 분할, 초록색은 합병을 나타낸다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Merge_sort_algorithm_diagram.svg/1280px-Merge_sort_algorithm_diagram.svg.png){: w="60%"}
_Fig 1. A recursive merge sort (top-down).  
(Image source: Wikipedia. [Merge sort](https://en.wikipedia.org/wiki/Merge_sort#/media/File:Merge_sort_algorithm_diagram.svg).)_  

## Algorithm

알고리즘을 작성하기 전에 아래와 같은 요소를 고려해야 한다.

먼저, 서브 리스트를 분할하기 위해 필요한 것은 서브 리스트의 시작과 끝 인덱스를 나타내는 $l$와 $r$이다. 어떤 서브 리스트를 2개의 하위 서브 리스트로 분할할 때 $l$과 $r$의 가운데 값 $m = \dfrac{l + r}{2}$울 사용한다. 왼쪽 서브 리스트는 $l$부터 $m$, 오른쪽 서브 리스트는 $m + 1$부터 $r$로 분할된다.

두 번째로, 합병 시 필요한 것은 임시 리스트이다. 두 서브 리스트를 합병할 때 원래 리스트에 합병하는 것이 아니라, 임시 배열에 합병한다. 그렇게 해서 최종 합병이 끝나면 임시 배열은 정렬이 완료된 상태가 된다. 이를 원래 배열에 복사한다.

자, 이제 알고리즘을 작성해보자. $\mathbf{a}_i$는 `a[i]`와 동일한 의미이다.

> ##### $\text{Algorithm: Merge sort (top-down)}$  
> $$
> \begin{align*}
> & \textstyle \text{Input: an array $\mathbf{a} \in \mathbb{R}^n$ to sort, a temporary array $\mathbf{b} \in \mathbb{R}^n$, the number of elements $n$} \\
> & \textstyle \text{$l$ is the left begin index, $r$ is the right end index} \\
> \\
> & \textstyle l \leftarrow 0 \\
> & \textstyle r \leftarrow n - 1 \\
> & \textstyle \text{split_merge($\mathbf{a}$, $\mathbf{b}$, $l$, $r$)} \\
> \\
> & \textstyle \text{function split_merge($\mathbf{a}$, $\mathbf{b}$, $l$, $r$):} \\
> & \textstyle \qquad \text{If $l < r$, then:} \\
> & \textstyle \qquad\qquad m \leftarrow \lfloor (l + r) \div 2 \rfloor \\
> & \textstyle \qquad\qquad \text{split_merge($\mathbf{a}$, $\mathbf{b}$, $l$, $m$)} \\
> & \textstyle \qquad\qquad \text{split_merge($\mathbf{a}$, $\mathbf{b}$, $m+1$, $r$)} \\
> & \textstyle \qquad\qquad \text{merge($\mathbf{a}$, $\mathbf{b}$, $l$, $m$, $r$)} \\
> & \textstyle \text{end} \\
> \\
> & \textstyle \text{function merge($\mathbf{a}$, $\mathbf{b}$, $l$, $m$, $r$):} \\
> & \textstyle \qquad i \leftarrow l \\
> & \textstyle \qquad j \leftarrow m + 1 \\
> & \textstyle \qquad \text{Loop for $k = l, l + 1, \dots$:} \\
> & \textstyle \qquad|\qquad \text{If $i \leq m$ and ($j > r$ or $\mathbf{a}_i \leq \mathbf{a_j}$), then:} \\
> & \textstyle \qquad|\qquad\qquad \mathbf{b}_k \leftarrow \mathbf{a}_i \\
> & \textstyle \qquad|\qquad\qquad i \leftarrow i + 1 \\
> & \textstyle \qquad|\qquad \text{else:} \\
> & \textstyle \qquad|\qquad\qquad \mathbf{b}_k \leftarrow \mathbf{a}_j \\
> & \textstyle \qquad|\qquad\qquad j \leftarrow j + 1 \\
> & \textstyle \qquad \text{until $k = r$} \\
> & \textstyle \qquad \text{copy $\mathbf{b}_{l:r+1}$ to $\mathbf{a}_{l:r+1}$} \qquad \text{($l:r+1$ is $l, l+1, \dots, r$)} \\
> & \textstyle \text{end} \\
> \end{align*}
> $$

## C++ Code

이제 위 알고리즘을 바탕으로 C++ 코드를 작성해보자. 위 알고리즘과 거의 동일하다. `dtype`은 임의의 비교 가능한 데이터 타입이다.

```c++
void merge(dtype a[], dtype b[], int l, int m, int r)
{
    // merge to b
    int i = l;
    int j = m + 1;
    for (int k = l; k <= r; k++)
    {
        if (i <= m && (j > r || a[i] <= a[j]))
        {
            b[k] = a[i];
            i++;
        }
        else
        {
            b[k] = a[j];
            j++;
        }
    }

    // copy merged part of b to a
    for (int k = l; k <= r; k++)
    {
        a[k] = b[k];
    }
}

void split_merge(dtype a[], dtype b[], int l, int r)
{
    if (l >= r)
        return;

    int m = (l + r) / 2;
    split_merge(a, b, l, m);
    split_merge(a, b, m + 1, r);
    merge(a, b, l, m, r);
}

void merge_sort(dtype a[], dtype b[], int n)
{
    split_merge(a, b, 0, n - 1);
}
```

## References

[1] Wikipedia. [Merge sort](https://en.wikipedia.org/wiki/Merge_sort).