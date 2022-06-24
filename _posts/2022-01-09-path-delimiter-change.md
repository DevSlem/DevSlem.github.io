---
title: "[WinForm] 경로 구분자 변경 프로그램"
excerpt: "경로 구분자(Delimiter)를 \"/ -> \\ 또는 \\ -> /\" 로 변경해주는 프로그램"
categories:
    - C# Program
tags:
    - [경로, 구분자]
date: 2022-01-09
last_modified_at: 2022-01-09
---

# 1. 개요

Visual Studio Code로 마크다운을 작성하다보면 이미지 포함 로컬 경로를 삽입하는 경우가 상당히 많다. 
몰론 직접 `/assets/images/*` 등의 경로를 직접 텍스틀 쳐도, VS Code의 인텔리센스가 경로에 있는 파일을 추적해주기 때문에 직접 텍스트를 입력해도 괜찮았다.  
문제는 `.webp` 파일에서 발생했다. 왠지는 모르겠는데 `.webp` 파일의 경우 VS Code의 인텔리센스가 경로를 입력해도 파일을 추적하지 못한다.
따라서 직접 모두 입력하거나 경로를 복사해 붙여넣기 할 수 밖에 없다.  
그런데 여기서 또 문제가 발생한다. 다행히 VS Code에는 **Copy Relative Path** 기능이 있어서 로컬 경로를 쉽게 복사할 수 있지만, 경로 구분자(Delimiter)가 `/`가 아닌 `\`로 복사한다.
몰론 `\` 구분자를 사용해도 블로그에서는 이미지가 제대로 입력되지만 VS Code의 마크다운 프리뷰 기능에서는 구분자가 `/`가 아니면 이미지를 불러오질 못한다.  
`Ruby`를 통해 로컬에서 블로그 포스트를 확인하는 것도 나름 편리하지만, 어쨋든 VS Code에서 웹 브라우저로 옮겨 `F5` 키를 눌러 새로고침을 해야하는 불편함이 있다.  
반대로, 마크다운 프리뷰 기능은 따로 무언갈 하지 않아도 즉시 출력해주기 때문에 조금 더 편리하다. 그래서 어쩔 수 없이 경로 구분자의 형태를 변경해주는 기능이 필요했다.  
구글에 아무리 검색해도 관련 프로그램이 안나와서 그냥 내가 직접 프로그램을 만들었다. 
몰론, 이거 하나 때문에 프로그램까지 만드는게 오히려 비효율적일 수도 있지만 개발자의 욕구 아니겠는가? 그래서 실행에 옮겼다.



# 2. 프로그램

## 다운로드

아래 **경로 구분자 변경 프로그램**을 클릭하세요.

> **Download(다운로드):** [**경로 구분자 변경 프로그램**](/assets/programs/ChangePathDelimiter.zip)

<!--
> Download(다운로드): <a href="/assets/programs/ChangePathDelimiter.zip" download="ChangePathDelimiter.zip">경로 구분자 변경 프로그램</a>
-->


## 프로그램 기능 설명

사실 나는 UI 제작 및 배치에 대해 거의 모른다. UI 쪽은 아예 관심이 없다시피해서 그냥 사용만 가능한 정도로 UI를 배치했다. 사실 처음에는 콘솔로 만들려했는데 아무리 생각해도 이건 아닌거 같아서 **Windows Forms**로 만들었다.

### 프로그램 초기 실행 화면

프로그램 초기 실행 화면이다.  
기본 모드는 **\\(Backslash) -> /(Slash)**, **클립보드에 복사 켜기**이다.

![실행화면](/assets/images/path-delimiter-change(1).png)

### Backslash -> Slash

경로 구분자 `\`(Backslash)를 `/`(Slash)로 변경해준다.

![입력](/assets/images/path-delimiter-change(2).png)


입력 후 `Enter`키를 입력하면, 변경된 경로를 출력란에 출력하고 클립보드에 자동으로 복사해준다.

![출력1](/assets/images/path-delimiter-change(3).png)


### Slash -> Backslash

마찬가지의 방법으로 입력하면 된다.

![출력2](/assets/images/path-delimiter-change(4).png)




# 3. 알고리즘

알고리즘은 초간단하다. 딱히 설명하지 않아도 될 정도여서 코드만 올리겠다.

```cs
using System;
using System.Drawing;
using System.Windows.Forms;

namespace ChangePathDelimiter
{
    public partial class Form1 : Form
    {
        private Size lastSize;

        public Form1()
        {
            InitializeComponent();
            textBox1.Focus(); // 프로그램 실행 시 즉시 입력 텍스트 박스에 포커스함.
            lastSize = this.Size;
        }

        // KeyDown이 아닌 KeyPress 이벤트를 사용한 이유는 Enter키를 누를 시 윈도우 에러 알림 소리가 발생해서 이를 방지하기 위해서임.
        private void textBox1_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar == (char)Keys.Enter)
            {
                e.Handled = true; // 에러 알림 소리 방지
                if (backslashBtn.Checked)
                {
                    textBox2.Text = textBox1.Text.Replace('\\', '/');
                }
                else
                {
                    textBox2.Text = textBox1.Text.Replace('/', '\\');
                }

                // 클립보드에 복사
                if (copyOn.Checked)
                {
                    Clipboard.SetText(textBox2.Text);
                }
            }
        }

        private void Form1_SizeChanged(object sender, EventArgs e)
        {
            // 메인 윈도우의 크기가 변경되는 정도에 따라 입력, 출력 텍스트 박스의 크기를 변경함.
            Size changed = this.Size - lastSize;

            textBox1.Size += changed;
            textBox2.Size += changed;

            lastSize = this.Size;
        }
    }
}
```