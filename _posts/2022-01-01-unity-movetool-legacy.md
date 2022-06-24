---
title: "[Unity] Scene View에서 Vector의 Position 조작하기 - Deprecated"
excerpt: "이 포스트는 Legacy code를 소개하고 있기 때문에 새롭게 작성한 포스트를 참조해주세요."
categories:
    - Unity Algorithm
tags:
    - [Unity, Attribute, Reflection, Vector, Scene View]
date: 2022-01-01
last_modified_at: 2022-01-17
---

# Legacy Version

## 새롭게 작성된 포스트 Link

**이 포스트는 Legacy code를 소개하고 있습니다.** 아래에 새롭게 리뉴얼된 코드를 소개하는 포스트의 링크를 첨부하였으니 아래 링크의 포스트를 참조해주세요.  
아래 링크가 첨부된 포스트에서 소개하는 코드는 이 포스트에서 소개한 코드에서 발생한 버그가 수정되었으며, 기능이 추가된 업그레이드 버전입니다.

> [**New Version Post Link**](https://kgmslem.github.io/unity%20algorithm/unity-movetool-release1/)

## 버그 리포팅

기능 자체는 정상적으로 동작하지만, 아래와 같은 치명적인 버그가 발견되어 이 코드는 더이상 사용되지 않습니다. 또한 기존보다 더 기능이 업그레이드되었습니다.

![버그](/assets/images/serialized-object-exception.png)

Unity 공식 문서 [**Editor.serializedObject**](https://docs.unity3d.com/ScriptReference/Editor-serializedObject.html)의 설명

> Do not use the serializedObject inside OnSceneGUI or OnPreviewGUI. **Use the target property directly** in those callback functions instead.

<br>

# 1. 개요

유니티로 게임을 만들다 보면 Vector의 Position을 조작하는 일이 수도 없이 많다. 게임 오브젝트의 Position은 `Transform` 컴포넌트가 지원하는 자체적인 **Move Tool** 도구로 Scene View에서 쉽게 위치를 조작할 수 있지만, 일반적인 `Vector2` 혹은 `Vector3` 타입의 필드의 값을 변경할 때는 수동으로 값을 직접 입력해야 한다. 이는 굉장한 불편함이다. 몰론 Vector 구조체가 아닌, Empty GameObject를 생성 후 `Transform` 타입의 필드에 할당하여 조작하면 되지만 Vector 구조체 단독으로 사용하는 것보다는 메모리, 성능 등 효율성이 다소 떨어진다. 구조체의 성능적인 장점을 최대한 활용하는 것이 아무래도 좋지 않겠는가? 아래 장면을 보자.  

![조작](/assets/images/vector-movetool-control.webp)

위 장면을 보면 알 수 있지만 `Transfrom` 컴포넌트를 조작하는 것이 아닌 `Vector3` 타입의 직렬화된 필드 자체를 Scene View에서 조작하고 있다. 위와 같이 필드로 선언되어있는 Vector 구조체를 Scene View의 **Move Tool** 도구를 활용해 조작하는 방법을 알아보자.

<br>

# 2. 구현에 앞서 필요한 지식

우리가 흔히 유니티에서 인스펙터를 간편히 조작하기 위해 아래와 같이 필드에 Attribute를 할당한다. 

```csharp
[SerializeField]
private Vector3 privateVector;
```

<br>
위 코드에서 할당한 Attribute는 유니티에서 가장 자주 쓰이는 Attribute 중 하나인 `SerializeField`이다. 이 Attribute는 `private` 필드를 인스펙터 등에서 조작 가능하게 직렬화 시켜준다. 이처럼 Vector 구조체 필드 역시 복잡한 과정 없이, 간단히 Attribute의 선언만으로 쉽게 Scene View에서 조작하고 싶다. 이를 위해 위와 같이 Attribute를 만들기로 결정했다.

**Attribute(특성)**를 어떻게 구현할까? 나 역시 현재 Attribute에 대한 완벽한 이해가 있는 것은 아니다. MS 공식 문서 [**특성(C#)**](https://docs.microsoft.com/ko-kr/dotnet/csharp/programming-guide/concepts/attributes/)에서는 아래와 같이 기술하고 있다.

> 특성은 메타데이터 또는 선언적 정보를 코드(어셈블리, 형식, 메서드, 속성 등)에 연결하는 강력한 방법을 제공합니다. 특성이 프로그램 엔터티와 연결되면 _리플렉션_ 이라는 기법을 사용하여 런타임에 특성이 쿼리될 수 있습니다. 자세한 내용은 [**리플렉션(C#)**](https://docs.microsoft.com/ko-kr/dotnet/csharp/programming-guide/concepts/reflection)을 참조하세요.

위 내용만 봐서는 프로그래밍에 깊은 지식과 이해가 있지 않은 한 이해하기 어렵다고 생각한다. 또한 **Reflection(리플렉션)**이라는 개념도 등장한다. 이들은 C#의 고급기술에 속한다(뇌피셜). 본인도 아직 학부생이기 때문에 이 문서를 읽으면서 상당히 어려웠고, 아직 깊은 이해는 못했다. 어차피 우리의 목표는 Attribute에 대한 이해가 아닌, Vector의 Scene View에서의 조작이다. 따라서 Attribute와 Reflection에 대해서는 알고리즘 구현에 필요한 최소한만 소개할 것이다.


## Atribute(특성)

먼저, **Attribute**에 대해 내가 이해한 핵심적인 요약은 아래와 같다.

> Attribute는 **프로그램이 이해할 수 있는 주석**이다.

우리가 주석을 작성하는 이유는 개발하는 나, 코드를 읽는 다른 개발자가 코드의 내용을 이해할 수 있게 하기 위해 작성한다. Attribute 역시 그러하다. 한글로 번역된 **특성**이라는 단어가 굉장히 직관적이다. 프로그램에게 이 클래스, 인터페이스, 메서드, 프로퍼티 혹은 필드가 이러한 **특성**을 지니고 있다라고 알려주는 역할을 한다.  
예를 들면 C#에서 가장 유명한 Attribute 중 하나인 `Obsolete`를 보자. 이 **특성**은 프로그램 혹은 컴파일러에게 **더 이상 사용되지 않는다**라고 알려준다.  
마찬가지로 우리가 Vector 구조체를 Scene View에서 조작할 수 있게 하기 위해 Attribute를 만들려고 한 이유도 이와 같다. 즉, 이 필드는 **Scene View에서 Position을 조작할 수 있는 Move Tool 도구를 지원**해야한다고 프로그램에게 알리기 위해서이다.


## Reflection(리플렉션)

그렇다면 누가 어떤 Attribute를 가지고 있는지 알 수 있을까? C#에서는 **Reflection(리플렉션)**이라는 기법을 통해 가능하다. Reflection은 **런타임**에 **어떤 타입에 대한 정보를 뜯어볼 수 있도록 해준다**. 이는 굉장한 기능이다. 우리가 선언한 클래스 혹은 인터페이스를 비롯해 각종 메서드, 필드에 대해 런타임에 확인하고 조작할 수 있다는 것을 의미한다. 어떤 클래스 내에 선언한 필드에 어떤 Attribute가 할당되어있는지 역시 런타임에 조사가 가능하다.  
이것이 가능한 이유는 바로 아래와 같다.  

> C#의 모든 타입의 Base Class인 `Object` Class에 `Type GetType()` 메서드가 선언되어 있다.  
> 즉, 모든 형식의 타입에 대해 정보를 열람할 수 있다.

앞서 말했듯이 리플렉션은 **어떤 타입에 대한 정보를 뜯어볼 수 있도록 해준다**. 즉, 어떤 타입에 대한 데이터를 담고 있는 타입이 필요한데, 그게 바로 `System.Type` 클래스이다. `Type` 인스턴스를 통해 어떤 타입의 정보를 열람하고, 타입 멤버의 데이터를 얻거나 수정이 가능하다. 자세한 내용은 [**리플렉션(C#)**](https://docs.microsoft.com/ko-kr/dotnet/csharp/programming-guide/concepts/reflection)을 참조하기 바란다.

<br>

# 3. 전체 알고리즘

위에서 구현을 위한 지식을 간단히 소개했다. 필요한 문법은 아래에서 간단히 소개할 것이나, 자세한 내용이 궁금하다면 직접 문서를 참조하길 권장한다. 참고로 구현을 위한 코드와 아이디어는 [**ProGM/DraggablePoint.cs**](https://gist.github.com/ProGM/226204b2a7f99998d84d755ffa1fb39a)에서 얻었다. 나는 여기서 소개한 코드를 조금 더 많은 도구 지원과 효율성을 위해 재구성했다. 완성본을 즉시 사용하고 싶다면 아래 링크를 타고 들어가 복사해서 사용하면 된다.

> 전체 알고리즘 링크: [**kgmslem/MoveToolAttribute.cs**](https://gist.github.com/kgmslem/be107adb92f13b77533e5fc2c5196e91)

<br>

# 4. MoveToolAttribute

먼저, 필요한 Attribute를 선언하자. 나는 Scene View에서 **Move Tool** 도구 지원을 해준다는 의미로 `MoveToolAttribute`라는 이름으로 선언했다. 

<script src="https://gist.github.com/kgmslem/3f5c98c977d4f88b3212d70c18a5d812.js"></script>

<br>
C#에서 Attribute(특성)를 생성하기 위해서는 `System.Attribute` 클래스의 상속을 받아야 한다. 위 코드에서는 `UnityEngine.PropertyAttribute`를 상속받았는데 이 클래스는 `Attribute`클래스의 파생 클래스이다. 따라서 Attribute를 만들 수 있다.  

Attribute를 만들 때 클래스 명에 접미사로 `ClassNameAttribute`와 같이 접미사 `Attribute`를 넣는걸 MS 공식 문서에서는 권장하고 있다. `Attribute` 접미사를 넣어도 실제 특성을 사용할 때에는 `[ClassName]`과 같이 접미사 `Attribute`를 생략한 채 사용할 수 있다.

아래 코드를 보자.

```csharp
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = true)]
```

<br>
위 코드는 우리가 만들려는 Attribute의 특성을 기술하는 `AttributeUsage` 특성이다.  
`System.AttributeTargets`은 C#에 미리 정의되어 있는 `enum` 타입이다. 특성을 적용하는 데 유효한 애플리케이션 요소(클래스, 구조체, 메서드, 필드 등)를 지정할 수 있다. 참고로 `AttributeTargets`의 열거형 값은 `AttributeTargets.Class | AttributeTargets.Method`와 같이 **비트 OR 연산으로 결합** 하여 사용할 수 있다.  
우리는 Vector 타입의 **필드**에만 적용할 거기 때문에 `AttributeUsage`의 생성자에 `AttributeTargets.Field` 값을 인자로 주었다. 이에 대한 자세한 내용은 MS 공식 문서 [**AttributeTargets 열거형**](https://docs.microsoft.com/ko-kr/dotnet/api/system.attributetargets?view=net-6.0)에서 확인하기 바란다.


위 `MoveToolAttribute`에 선언한 프로퍼티는 다음과 같은 데이터를 담는다.

* `bool LocalMode` - 조작하려는 벡터를 로컬 좌표 모드로 실행할 지 여부(기본 모드: World Mode)
* `bool LabelOn` - Scene View에 레이블을 표시할 지 여부(기본: 표시함)
* `string Label` - 사용자 정의 레이블(기본 표시 레이블: 인스펙터에 표시되는 변수 명)

`MoveTool` Attribute는 SceneView에 **Move Tool**을 표시하겠다고 알리는 역할이다. 그 때 어떤 방식으로 표시할 지에 대해서는 위와 같은 프로퍼티 정보를 활용한다.

참고로 위 `string Label` 프로퍼티의 경우 기본값은 `string.Empty`이다. 이 이유는 이 특성이 어떤 변수에 할당되어있는지를 현재는 모르기 때문이다. 런타임에 `MoveToolAttribute`가 할당되어 있는 필드를 조사해 그 때 필드(변수) 이름을 적용하기 위해 기본값은 `string.Empty`로 초기화했다.

<br>

# 5. MoveToolEditor

`MoveTool` Attribute를 만들었으니 이제 이를 활용하여 Scene View에서 **Move Tool**을 표시하고 조작하는 기능을 구현하자. 여기서는 코드를 차근차근 보이면서 설명하겠다.

## MoveToolEditor 선언

먼저, **Move Tool**을 구현하기 위해 필요한 네임스페이스와 클래스 선언부이다.

```csharp
using System.Reflection;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(MonoBehaviour), true)]
public class MoveToolEditor : Editor {  }
```

<br>
Scene View를 조작하기 위해서는 `Editor` 클래스를 상속 받은 커스텀 에디터를 만들어야 한다. 이 때 커스텀 에디터를 만들기 위해서는 `CustomEditor` Atrribute를 부여해야 한다.  
`CustomEditor` Attribute의 생성자는 아래와 같다.

```csharp
public CustomEditor(Type inspectedType, bool editorForChildClasses);
```

<br>
`inspectedType`은 유니티 에디터에 표시할 타입, `editorForChildClasses`는 `inspectedType` 매개변수에 값을 준 타입의 상속을 받은 파생 클래스에서도 유니티 에디터에 표시할 것인지 여부를 묻는 매개변수이다.  
우리는 매개변수로 `typeof(MonoBehaviour)`를 주었는데 이는 모든 컴포넌트에서 **Move Tool** 기능을 사용하기 위해서이다. 모든 컴포넌트는 `MonoBehaviour`의 상속을 받기 때문에 `editorForChildClasses`에 `true`값을 주어야 한다.  


## MoveToolEditor 구조

구조는 크게 아래와 같이 만들었다.

* `GUIStyle style` - **Move Tool**에서 표시할 레이블의 스타일 인스턴스
* `void OnEnable()` - 유니티 이벤트 메서드로 `style` 인스턴스 속성 초기화
* `void OnSceneGUI()` - 유니티 이벤트 메서드로 Scene View를 조작함
* `bool IsVector()` - `SerializedProperty`가 `Vector2` 혹은 `Vector3`인지 체크
* `bool HasMoveToolAttribute()` - 위에서 정의한 `MoveTool` 특성을 갖고 있는지 체크
* `void SetPositionHandle()` - **Move Tool**을 Scene 내에 배치 및 활성화

위 기능들을 차근차근 구현해보자.

## 레이블 스타일 초기화

간단한 초기화 작업이기 때문에 코드만 적겠다.

```csharp
// 레이블 표시를 위한 GUIStyle 도구
private readonly GUIStyle style = new GUIStyle();

private void OnEnable()
{
    style.fontStyle = FontStyle.Bold;
    style.normal.textColor = Color.white;
}
```

## Vector 여부 체크

먼저, Editor에서 다루기 위해서는 직렬화가 가능해야 한다. 유니티에서는 직렬화 가능한 프로퍼티(참고로 여기서 말하는 프로퍼티는 C#의 프로퍼티가 아닌 에디터에서 제어할 수 있는 속성을 의미한다)를 `SerializedProperty` 타입의 인스턴스로 제공한다. 단순 Vector 타입의 필드 뿐만 아니라 컬렉션도 지원할 계획이기 때문에 우리는 크게 2개로 나눠서 코딩해야 한다.

1. Vector 필드
2. Vector 타입의 컬렉션 필드

유니티에서 직렬화 가능한 컬렉션(배열, 리스트 등)은 `SerializedProperty.isArray`를 통해 확인할 수 있다. 일반 필드일 경우 `SerializedProperty.propertyType`을 통해 체크하면 되지만, 컬렉션의 경우 `SerializedProperty.arrayElementType`을 통해 체크해야 한다. 몰론, `SerializedProperty.propertyType`로도 체크할 수는 있지만 다소 귀찮다. 따라서 실제 대상 타입이 Vector인지 체크하기 위해 위 두 프로퍼티의 리턴 타입에 맞춰 아래와 같이 2개의 메서드를 정의했다.

* `SerializedProperty.arrayElementType`의 리턴 타입: `string`
* `SerializedProperty.propertyType`의 리턴 타입: `SerializedPropertyType`

```csharp
// Vector 타입 여부를 문자열로 체크
private bool IsVector(string typeInfo) => typeInfo == typeof(Vector2).Name || typeInfo == typeof(Vector3).Name;

// Vector 타입 여부를 SerializedPropertyType enum 타입으로 체크
private bool IsVector(SerializedPropertyType typeInfo) => typeInfo == SerializedPropertyType.Vector2 || typeInfo == SerializedPropertyType.Vector3;
```


## MoveTool Attribute 소유 여부 체크

모든 Vector 필드에서 **Move Tool** 기능이 작동하면 안된다. 개발자가 원하는 Vector 필드에서만 **Move Tool** 기능이 동작해야하고 이를 위해 우리가 앞서 만든 `MoveTool` Attribute를 필드에 할당했는지 조사한다.

```csharp
private bool HasMoveToolAttribute(SerializedProperty property, out MoveToolAttribute attr) {  }
```

<br>
`property` 매개변수에 `MoveTool` 특성이 할당되어있는지 조사하고, 특성 데이터를 다른 곳에서 활용할 수 있게 `MoveToolAttribute` 인스턴스를 반환해준다.

먼저, 프로퍼티가 원래 위치한 클래스 필드의 정보를 리플렉션을 통해 불러온다.

```csharp
attr = null;

// SerializedProperty의 이름에 해당하는 필드 정보를 가져오는데 실패 시 false 반환
var field = serializedObject.targetObject.GetType().GetField(property.name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
if (field == null)
    return false;
```

<br>
`serializedObject`는 `Editor` Base 클래스에있는 멤버로 인스펙터에 표시되는 직렬화된 오브젝트 정보를 포함한다. 우리는 실제 유니티 오브젝트의 컴포넌트 정보가 필요하기 때문에 `serializedObject.targetObject.GetType()`을 통해 실제 컴포넌트의 타입 정보를 가져온다.  

`Type` 클래스에는 `FieldInfo` 인스턴스를 반환하는 `GetField()`멤버가 존재한다. 이는 어떤 타입에 선언되어있는 특정 필드 정보를 반환하는 메서드이다. 매개변수로 필드의 이름과 `BindingFlags` 정보를 받는다. 필드에는 `public`, `private`, `static` 등 여러 키워드가 붙는다. `BindingFlags` `enum`타입은 검색하려는 필드의 속성을 필터링 시킬 수 있다.  

위 내용을 바탕으로 가져오려는 필드를 필터링 했다. `SerializedProperty.name`은 실제 직렬화된 프로퍼티의 이름이기 때문에 우리가 가져오려는 필드 이름과 동일하다. **Move Tool**을 표시할 필드는 인스턴스 멤버여야하고, 액세스 한정자와 관계 없이 `SerializedField`라면 모두 가져온다. 이를 위해 `BindingFlags`의 값을 다음과 같이 결합시켰다.

> `BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic`

이 때 필드 정보를 가져오는데 실패한다면 `false`를 반환하고 `attr` 매개변수에는 `null`을 반환한다.

이제, 가져온 `FieldInfo` 정보를 바탕으로 `MoveTool` Attribute가 할당되어있는지 조사한다. `FieldInfo.GetCustomAttributes()` 메서드를 통해 원하는 특성을 가져올 수 있다.

```csharp
// 가져온 필드에 MoveTool Attribute가 할당되어있는지 체크
var moveToolAttr = field.GetCustomAttributes(typeof(MoveToolAttribute), false);
if (moveToolAttr.Length > 0)
{
    attr = moveToolAttr[0] as MoveToolAttribute;
    return true;
}
else
{
    return false;
}
```

<br>
완성된 메서드는 아래와 같다.

```csharp
// property에 MoveTool Attribute가 할당되어있는지 체크와 동시에 MoveToolAttribute 인스턴스 반환
private bool HasMoveToolAttribute(SerializedProperty property, out MoveToolAttribute attr)
{
    attr = null;

    // SerializedProperty의 이름에 해당하는 필드 정보를 가져오는데 실패 시 false 반환
    var field = serializedObject.targetObject.GetType().GetField(property.name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
    if (field == null)
        return false;


    // 가져온 필드에 MoveTool Attribute가 할당되어있는지 체크
    var moveToolAttr = field.GetCustomAttributes(typeof(MoveToolAttribute), false);
    if (moveToolAttr.Length > 0)
    {
        attr = moveToolAttr[0] as MoveToolAttribute;
        return true;
    }
    else
    {
        return false;
    }
}
```


## Position Handle 생성 및 배치 메서드 구현

`Handles.PositionHandle()` 메서드를 사용하면 Scene View에 벡터를 조작할 수 있는 Position Handle을 생성 및 배치할 수 있다.  
현재 `SerializedProperty`가 `Vector3`일 때와 `Vector2`일 때 참조해야하는 멤버가 다르기 때문에 따로 처리했다.

```csharp
// Scene View에 Position Handle을 생성 및 배치함
private void SetPositionHandle(SerializedProperty property, string labelText, bool localMode)
{
    // 로컬 모드일 경우 기준 좌표를 원점이 아닌 현재 target 오브젝트의 위치로 설정
    Vector3 center = localMode ? (target as MonoBehaviour).transform.position : Vector3.zero;

    switch (property.propertyType)
    {
        case SerializedPropertyType.Vector3:
            Handles.Label(center + property.vector3Value, labelText, style); // 레이블 할당
            property.vector3Value = Handles.PositionHandle(center + property.vector3Value, Quaternion.identity) - center; // 핸들 배치
            serializedObject.ApplyModifiedProperties();
            break;
        case SerializedPropertyType.Vector2:
            Handles.Label((Vector2)center + property.vector2Value, labelText, style); // 레이블 할당
            property.vector2Value = Handles.PositionHandle((Vector2)center + property.vector2Value, Quaternion.identity) - center; // 핸들 배치
            serializedObject.ApplyModifiedProperties();
            break;
    }
}
```


## 위에서 구현한 기능들을 조합해 Move Tool 구현

이제 위 기능들을 조합해 Scene View를 조작하기 위해 아래 `OnSceneGUI()` 유니티 이벤트 메서드를 선언하자.

```csharp
public void OnSceneGUI() {  }
```

<br>
직렬화된 프로퍼티를 가져오기 위해 아래와 같은 코드를 작성한다.

```csharp
var property = serializedObject.GetIterator();
while (property.Next(true))
{

}
```

<br>
`GetIterator()`는 첫번째 직렬화된 프로퍼티를 가져오고, 다음 직렬화된 프로퍼티를 가져오기 위해서는 `Next()` 메서드를 호출한다. 직렬화된 오브젝트에 속해있는 직렬화된 프로퍼티를 모두 가져올 때 까지 기능을 작동시킨다.  
이 때 `Next()` 매개변수에 `false`를 주면 **Exception**이 발생한다. 유니티 공식 문서에는 배열, 구조체 등의 복합 타입?(공식 문서에서는 **a compound type**이라고 명칭하고 있다)에서는 위 매개변수 `enterChildren` 값에 따라 중첩된(nested) 프로퍼티들을 방문하거나, 이 복합 타입 직후의 프로퍼티로 스킵할 지 여부를 결정한다고는 적혀있는데 무슨 소린지 모르겠다. 자세한 내용은 유니티 공식 문서 [**SerializedProperty.Next**](https://docs.unity.cn/2022.1/Documentation/ScriptReference/SerializedProperty.Next.html)에 적혀있으니 아는 분 있으면 알려주시면 좋겠다.  

이제 `while`문 내부를 구현하자.  
앞서 [**Vector 여부 체크**](#vector-여부-체크)에서 Vector 타입의 필드와 컬렉션 필드를 나눠서 처리해야 한다고 언급했었다. 이를 반영해 다음과 같이 처리했다.

```csharp
// 현재 직렬화된 프로퍼티가 컬렉션일 경우
if (property.isArray)
{
    // Vector가 아닌 경우 Skip
    // Vector 타입이지만 MoveTool Attribute가 없는 경우 Skip
    if (!IsVector(property.arrayElementType) || !HasMoveToolAttribute(property, out var attr))
        continue;

    for (int i = 0; i < property.arraySize; i++)
    {
        SerializedProperty element = property.GetArrayElementAtIndex(i);

        // Move Tool 도구를 설정
        SetPositionHandle(element, attr.LabelOn ?
            ((string.IsNullOrEmpty(attr.Label) ? property.name.InspectorLabel() : attr.Label) + $" [{i}]") : string.Empty, attr.LocalMode);
    }
}
// 컬렉션이 아닐 경우
else
{
    // Vector가 아닌 경우 Skip
    // Vector 타입이지만 MoveTool Attribute가 없는 경우 Skip
    if (!IsVector(property.propertyType) || !HasMoveToolAttribute(property, out var attr))
        continue;

    SetPositionHandle(property, attr.LabelOn ? 
        (string.IsNullOrEmpty(attr.Label) ? property.name.InspectorLabel() : attr.Label) : string.Empty, attr.LocalMode);

}
```

<br>
**컬렉션이 아닐 경우**부터 먼저 보자. `if`문에서 앞서 정의한 `IsVector()`와 `HasMoveToolAttribute()` 메서드를 통해 필터링을 한다. 필터링 이후 **Move Tool** 기능을 작동시킬 프로퍼티임이 확인되었다면 이제 Position Handle을 생성 및 배치하자. 여기서 우리가 앞서 정의한 `MoveTool` Attribute의 프로퍼티 정보를 반영한다. 코드를 컴팩트하게 작성하기 위해 다소 복잡한 형태로 호출했다.

**컬렉션일 경우**에는 직렬화된 컬렉션 프로퍼티에 속해있는 각 원소에 접근 후 각 원소마다 **Move Tool** 기능을 작동시킨다. 또한 레이블의 경우 구분을 위해 컬렉션의 인덱스를 추가해주었다.

참고로 위에서 사용한 `property.name.InspectorLabel()` 메서드는 내가 정의한 `string` 확장 메서드이다. 코드는 아래와 같다.

<script src="https://gist.github.com/kgmslem/d98841406eb4c261b34ae17c32acb241.js"></script>

<br>
완성된 `OnSceneGUI()` 메서드의 코드는 아래와 같다.

```csharp
public void OnSceneGUI()
{
    var property = serializedObject.GetIterator();
    while (property.Next(true))
    {
        // 현재 직렬화된 프로퍼티가 컬렉션일 경우
        if (property.isArray)
        {
            // Vector가 아닌 경우 Skip
            // Vector 타입이지만 MoveTool Attribute가 없는 경우 Skip
            if (!IsVector(property.arrayElementType) || !HasMoveToolAttribute(property, out var attr))
                continue;

            for (int i = 0; i < property.arraySize; i++)
            {
                SerializedProperty element = property.GetArrayElementAtIndex(i);

                // Move Tool 도구를 설정
                SetPositionHandle(element, attr.LabelOn ?
                    ((string.IsNullOrEmpty(attr.Label) ? property.name.InspectorLabel() : attr.Label) + $" [{i}]") : string.Empty, attr.LocalMode);
            }
        }
        // 컬렉션이 아닐 경우
        else
        {
            // Vector가 아닌 경우 Skip
            // Vector 타입이지만 MoveTool Attribute가 없는 경우 Skip
            if (!IsVector(property.propertyType) || !HasMoveToolAttribute(property, out var attr))
                continue;

            SetPositionHandle(property, attr.LabelOn ? 
                (string.IsNullOrEmpty(attr.Label) ? property.name.InspectorLabel() : attr.Label) : string.Empty, attr.LocalMode);

        }
    }
}
```

## MoveToolEditor 알고리즘

<script src="https://gist.github.com/kgmslem/7d6c5147917cef319f6040f651d48d19.js"></script>

<br>

# 6. MoveToolDrawer

사실 **Move Tool** 구현은 모두 끝났다. 지금 구현하려는 것은 조금 더 커스터마이징을 위한 선택적 요소이다. `MoveToolDrawer`의 역할은 인스펙터에 `MoveTool` Attribute를 사용하는 필드에 대한 추가적인 정보 표시를 위해 만들었다. **Move Tool** 기능만 필요하다면 이 부분은 건너뛰어도 된다.  

나는 `MoveTool` Attribute의 데이터 중 `LocalMode` 프로퍼티 값에 따라 인스펙터에 월드/로컬 좌표 모드 정보를 보여주고, 각 `Label` 프로퍼티를 활용해 인스펙터에 표시되는 필드 텍스트를 변경했다.


## MoveToolDrawer 알고리즘

<script src="https://gist.github.com/kgmslem/89e61669fcefc641bc7863c75025126d.js"></script>

<br>
간단히 소개만 하면 `PropertyDrawer` 베이스 클래스에서 `attribute`, `fieldInfo` 등 리플렉션을 위한 멤버를 제공한다. 이 때 `attribute` 프로퍼티의 경우 `PropertyAttribute` 타입이다. 따라서 커스텀 특성이 `PropertyAttribute`의 상속을 받아야 한다.  
또한 `OnGUI()` 메서드의 경우 컬렉션에 대해서는 컬렉션 필드 자체에 대해서는 실행되지 않고, 컬렉션 원소에 대해서 실행된다. 즉 `property` 매개변수의 Argument는 컬렉션 원소이며, 총 컬렉션 원소 개수만큼 실행된다. 따라서 어떤 컬렉션에 대해 1번만 처리하기 위해 몇가지 처리를 해주었다. 아무리 검색해도 컬렉션 자체에 대한 실행과 관련된 정보는 없어 일단 위와 같이 처리했다.

<br>

# 7. MoveToolTest - 활용

## 활용 코드 예시

`MoveTool` Attribute를 Vector 타입의 필드와 컬렉션 필드에 적용하는 간단한 코드이다.

```csharp
using System.Collections.Generic;
using UnityEngine;

public class MoveToolTest : MonoBehaviour
{
    [MoveTool]
    public Vector3 publicVector3;

    [MoveTool]
    public Vector2 publicVector2;

    [MoveTool, SerializeField]
    private Vector3 privateVector3;

    [MoveTool, SerializeField]
    private Vector2 privateVector2;

    [MoveTool(LocalMode = true)]
    public Vector3 localVector;

    [MoveTool]
    public Vector3[] vectorArray = new Vector3[3];

    [MoveTool]
    public List<Vector2> vectorList = new List<Vector2>();
}
```


## 에디터 장면

### Inspector

각 Vector 모두 정상적으로 Serialize되었으며 **Position Mode** 정보도 잘 표시하고 있다. 레이블 테스트는 현재 코드와 사진에는 없지만 직접 확인해보면 역시 정상 작동한다.

![인스펙터](/assets/images/vector-movetool-inspector.png)

### Scene View

위 Inspector와 비교했을 때 특히 유의미한 것은 `Local Vector`이다. `Local Vector` 필드의 경우 Position Mode가 Local Mode이기 때문에 좌표 값이 현재 컴포넌트가 포함된 오브젝트의 위치로부터 상대적으로 [-3, 1] 만큼 떨어진 위치에 정확히 위치하고 있다.

![씬뷰](/assets/images/vector-movetool-sceneview.png)