---
title: "[Unity] 유니티 첫 게임 - 첫번째 보스 리퍼"
excerpt: "Unity로 첫번째 보스 리퍼(Reaper)를 구현함"
categories:
    - Unity Game Development
tags:
    - [Unity, 유니티, Game, 게임, 유니티 개발, 유니티 프로젝트]
date: 2022-01-09
last_modified_at: 2022-01-09
---

# 1. 개요

유니티로 개발한 첫 게임에서 구현한 보스 몬스터이다.  
아직 개발 중이기 때문에 천천히 업데이트할 예정이다.  

## 에셋 정보

사용한 에셋은 동료가 직접 그린 그림이며 아래와 같다.

### 보스 기본 상태 - 무기 모드

![보스 무기 On](/assets/images/unity-my-first-game/reaper-boss-weapon-on.png)

### 보스 무기 - 낫

![보스 기본 무기](/assets/images/unity-my-first-game/reaper-boss-weapon-reaping-hook.png)




# 2. 낫 무기

## 낫 무기 구조

기본 구조는 아래와 같다.

* **Reaping Hook** - 낫 무기 본체  
    * **Effect** - 낫 무기 이동 흔적 이펙트
    * **Bar Start** - 막대기 부분의 시작점
    * **Bar End** - 막대기 부분의 끝점

위 **Bar Start**와 **Bar End** 오브젝트는 막대기 부분의 영역 표시를 위한 오브젝트로 칼날 부분과 막대기 부분의 데미지 경감을 주기 위해 만들었다.

### 낫 - Hierarchy

![낫 하이어아키](/assets/images/unity-my-first-game/reaper-boss-weapon-reaping-hook-hierarchy.png)

## 콜라이더 설정

낫 무기 이미지 자체가 `Capsule Collider 2D` 등의 일반적인 `Collider`로는 만들기 어렵다. `Collider`를 겹쳐서 만들어도 되지만 이는 성능적인 이슈가 있다고 한다. 따라서 이미지 형상 그대로 `Collider`를 형성할 수 있는 `Polygon Collider 2D`를 활용하였다.

### 낫 - Scene View

![낫 Scene View](/assets/images/unity-my-first-game/reaper-boss-weapon-reaping-hook-sceneview.png)


### 낫 - Inspector

![낫 인스펙터](/assets/images/unity-my-first-game/reaper-boss-weapon-reaping-hook-inspector.png)


## 이펙트 설정

`Trail Renderer` 컴포넌트를 활용해 낫의 이동 궤적을 따라 이펙트를 렌더링하는 기능을 추가했다.

![이펙트](/assets/images/unity-my-first-game/reaper-boss-weapon-reaping-hook-effect-inspector.png)




# 3. 낫 무기를 활용한 스킬

낫 무기의 형태를 완성했으니 이제 이를 스킬을 구현해 활용하겠다.

## Single Reaping Hook Swing

낫 하나를 타겟(플레이어) 근처에 생성해 휘두른다.

![Single 낫 스윙](/assets/images/unity-my-first-game/reaper-boss-single-reaping-hook-swing.webp)


## Double Reaping Hook Swing

낫 두개를 타겟(플레이어) 근처에 생성해 휘두른다.

![Double 낫 스윙](/assets/images/unity-my-first-game/reaper-boss-double-reaping-hook-swing.webp)

## Throw Rotated Reaping Hook

회전하는 낫을 리퍼 근처에 생성해 일정 시간 대기 후 플레이어를 향해 날린다.

![회전 낫 던지기](/assets/images/unity-my-first-game/reaper-boss-throw-rotated-reaping-hook.webp)

## Sequential Reaping Hook

낫 4개를 순차적으로 생성 후 순차적으로 일직선으로 날린다. 플레이어에 위치에 따라 회전과 스케일의 $x$성분을 조정한다.

![연속 낫](/assets/images/unity-my-first-game/reaper-boss-sequential-reaping-hook.webp)




# 4. 손뼈 무기를 활용한 스킬

손뼈 무기의 경우 특별히 따로 설명할 것은 없다. 따라서 바로 손뼈 무기를 활요한 스킬 설명으로 넘어가겠다.

## Grab

플레이어의 상당한 뒤쪽에서 손뼈를 생성 후, 손뼈가 바라보는 $x$축과 평행한 방향으로 빠르게 이동한다. 손뼈가 타겟(플레이어)과 충돌 시 플레이어를 잡는다. **잡힌 상태에서 플레이어는 이동과 총기 발사가 제한되며, 지속데미지를 입는다.** 잡힌 상태에서 빠져나오기 위해서는 **스페이스 키를 총 30번** 눌러야 한다.  
참고로 스페이스 키를 눌러야 하는 횟수 표시를 위한 UI를 적용하기 위해 `TextMeshPro - Text` 컴포넌트를 활용하였다.

![그랩](/assets/images/unity-my-first-game/reaper-boss-grab.webp)


## Smash

추가될 예정




# 5. 그 외 스킬

## Spread

이 스킬은 다른 개발자 동료가 개발한 스킬이다.  
투사체를 4방향으로 지속적으로 발사한다. 매 발사 시점마다 발사각을 회전시킨다.

![스프레드](/assets/images/unity-my-first-game/reaper-boss-spread.webp)