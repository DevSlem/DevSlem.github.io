---
title: "[Unity] Save System Pattern"
excerpt: "Save와 Load 작업을 조금 더 효율적으로 실행하기 위한 패턴을 소개한다."
categories:
    - Unity Algorithm
tags:
    - [Unity, Save]
date: 2022-04-06
last_modified_at: 2022-05-14
---

# Introduction

Unity에서 Save System을 효과적으로 구축하기 위한 패턴을 직접 생각하고 구현해보았다. 팀원들과 밑바닥부터 스스로 생각하고 논의해서 얻어낸 결과이기 때문에 지금 소개할 패턴은 오히려 비효율적일 수도 있다. 다만 우리의 목적에는 충분히 들어맞고, 직접 생각하고 구현해보았다는 의의가 있다고 생각한다.

# Goal

우리의 목적은 다음과 같다.

* 여러 개의 게임 데이터를 하나의 파일로 저장한다. 예를 들면 플레이어 데이터, 상점 데이터, 게임 진행 현황 등 분할된 게임 데이터들을 하나의 파일에 저장한다.
* 저장해야할 데이터는 각각의 인스턴스에서 관리한다.
* 저장해야할 데이터가 늘어나거나 줄어도 메인 시스템은 변경되지 않는다.
* Save나 Load를 요청하는 컨트롤러는 실제 어떤 데이터가 저장되고 Load되는지 모른다. 오직, Save, Load에 대한 시점만 결정한다.
* Load 명령이 요청되면 각각의 인스턴스는 자기가 저장했던 데이터를 획득해 동기화한다.

위 목적을 달성하기 위해 우리는 크게 역할을 3가지로 구분했다.

* Save 매니저(파일 입출력을 담당)
* Save 컨트롤러(Save, Load 시점을 결정함)
* Save 리스너(Save, Load가 발동 될 시 각자 원하는 데이터를 Save 및 Load함)

#### Save 매니저

Save 매니저는 파일 입출력을 담당하고 Save, Load에 대한 명령을 받으면 리스너에게 알린다.

#### Save 컨트롤러

Save 컨트롤러는 Save 매니저의 Save, Load 메서드를 호출함으로써 Save 매니저에게 명령한다. Save 시점은 수동일 수도 있고 자동일 수도 있으며 이는 각 컨트롤러 오브젝트에 따라 다르다.

#### Save 리스너

실제 저장되는 데이터를 관리한다. Save 매니저를 구독하며 Save 알림을 받으면 저장할 데이터를 반환한다. Load 알림을 받으면 저장했던 데이터를 입력 받아 동기화 작업을 처리한다. 예를 들면 Save 알림을 받으면 Player의 체력, 스태미너, 공격력 등 Player의 Stat 데이터를 모아놓은 데이터 컨테이너를 반환하고, Load 알림을 받으면 저장되어있던 체력, 스태미너, 공격력 등을 자신의 Stat 데이터와 동기화한다. Save나 Load하라는 알림을 받더라도, 어떤 데이터를 언제 Save하고 Load 할 지에 대해서는 전적으로 리스너가 결정한다. 


# Class Diagram

위 내용을 바탕으로 아래와 같이 클래스 다이어그램을 만들어보았다. **Save 컨트롤러**에 대해서는 따로 표시하진 않았다.

![](/assets/images/save-system-diagram.png)

먼저, 저장해야할 새로운 오브젝트가 추가된 경우 단순히 `SaveManager`에 `ISaveable` *interface*의 상속을 받은 **Listener**를 등록하고 데이터 컨테이너만 만들면 된다. 즉, 굉장히 유연해진다. 또한 기존에 등록된 오브젝트의 저장될 데이터 항목을 변경하는데도 굉장히 유연하다. 데이터 컨테이너는 전적으로 **Listener**가 관리하기 때문이다. 이는 오브젝트 제거에도 동일하게 적용된다.  
다만, 모든 데이터를 한번에 하나의 파일에 저장하고 싶진 않을 수 있다. 예를 들면 플레이어의 세팅(e.g. 입력키, 음량, 그래픽 등)과 플레이어의 게임 데이터 파일(실제 플레이어의 게임 진행 현황)은 완전히 다르다. 이것들을 한번에 저장하는 것은 말이 안되며 플레이어의 게임 데이터 파일을 여러개 만들고 싶을 수 있다. 이를 위해 우리는 2가지 요소를 도입했다. 첫번째는 `SaveKey`이다. 저장할 데이터를 큰 틀에서 분류하는 역할을 한다. 위 클래스 다이어그램을 보면 알겠지만 `SaveKey.GameData`와 `SaveKey.Setting`이 *enum* 타입의 값으로써 존재한다. 이는 데이터가 저장되는 단위를 큰 틀에서 분류한다. 두 번째는 같은 분류라도 여러 개의 파일을 만들고 싶을 수 있다. 이 때는 단순히 파일 이름인 `fileName`을 구분해주면 된다. 참고로 Key 값에 대한 타입은 굳이 *enum*이 아니여도 된다. 확실한건 큰틀에서 분류할 수 있는 역할만 수행해주면 된다.


# Source Code

위 **Save System**에 적용 된 내용을 전부 이 포스트에서 보이는건 무리가 있어 **Listener**에 대해서만 보이겠다.

## ISaveable interface

```c#
/// <summary>
/// Interface for listener to save and load.
/// </summary>
public interface ISaveable
{
    /// <summary>
    /// If multiple instances of same type subscribe save manager, you need to identify them by ID.
    /// </summary>
    string ID { get; }

    /// <summary>
    /// The listener can save a data by returning it. You need to define "Serializable" attribute for the data type.
    /// </summary>
    /// <returns>a data to save</returns>
    object Save();

    /// <summary>
    /// The listener can load a data by getting from data parameter.
    /// </summary>
    /// <param name="data">a data loaded</param>
    void Load(object loaded);
}
```

## Player Data Save Example

등록과 해제 시점은 각 클래스의 역할에 맞게 적절한 시점으로 설정해주면 된다.
여기서는 `PlayerEntity`가 존재하는 동안 항시 Save/Load 로직이 동작해야하기 때문에 `Awake()`에서 등록하고 `OnDestory()`에서 등록을 해제한다. 

```c#
public class PlayerEntity : MonoBehaviour, ISaveable
{
    private void Awake()
    {
        SaveManager.Add(this, SaveKey.GameData);
    }

    private void OnDestroy()
    {
        SaveManager.Remove(this, SaveKey.GameData);
    }

    string ISaveable.ID => null;

    object ISaveable.Save() => new PlayerData(this, this.IsDead ? this.MaxHealth : this.Health);

    void ISaveable.Load(object loaded)
    {
        if (loaded is PlayerData data)
        {
            this.MaxHealth = data.maxHelath;
            this.Health = data.health;
            this.MaxStamina = data.maxStamina;
            this.ResurrectionChance = data.resurrectionChance;
            this.RemainedLife = data.remainedLife;
        }
    }
}

[System.Serializable]
public class PlayerData
{
    public readonly float health;
    public readonly float maxHelath;
    public readonly float maxStamina;
    public readonly bool resurrectionChance;
    public readonly int remainedLife;

    public PlayerData(PlayerEntity player) : this(player, player.Health) { }

    public PlayerData(PlayerEntity player, float health)
    {
        this.health = health;
        this.maxHelath = player.MaxHealth;
        this.maxStamina = player.MaxStamina;
        this.resurrectionChance = player.ResurrectionChance;
        this.remainedLife = player.RemainedLife;
    }
}
```

# Feedback

현재는 `SaveManager`에서 Binary 형식의 파일만 입출력하고 있다. 현재 프로젝트에서는 Binary 형식의 파일 입출력이면 충분했지만 CSV, Json 등의 다양한 파일 형식으로 저장하고 싶을 수 있다. 따라서 이에 대한 개선이 필요하다.