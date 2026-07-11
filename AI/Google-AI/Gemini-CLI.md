

# [Gemini CLI](https://geminicli.com/)
1. Sign in with Google                                                                                                             
2. Use Gemini API Key                                                                                                             
3. Vertex AI   



---

## 1：Google 账号登录（推荐）✅
如果你**直接使用 Google 账号登录 Gemini CLI（OAuth 登录）**，**不会产生 API 或 Token 费用**。Google 给开发者提供了一套免费的配额，不按 token 计费。([Google Gemini](https://google-gemini.github.io/gemini-cli/docs/quota-and-pricing.html?utm_source=chatgpt.com "Gemini CLI: Quotas and Pricing | gemini-cli"))

安装 Gemini CLI 后：

```bash
gemini
```

第一次运行会提示：

```
Login with Google
```

登录你的 Google 账号即可。

这种模式：

- ✅ 不需要 API Key
    
- ✅ 不需要绑定信用卡
    
- ✅ 不按 token 收费
    
- ✅ 不会产生 Gemini API 账单
    

目前官方免费额度大约是：

- **1000 requests / 天**
    
- **60 requests / 分钟**
    
- 可使用 Gemini 模型家族（通常包括 Gemini 2.5 Pro）([Google Gemini](https://google-gemini.github.io/gemini-cli/docs/quota-and-pricing.html?utm_source=chatgpt.com "Gemini CLI: Quotas and Pricing | gemini-cli"))
    

这里限制的是 **请求次数（requests）**，而不是 token 数量。


根据你提供的信息，可以做出比较明确的判断。

### 你的账号状态

目前可以确认：

* ✅ 能正常访问 [https://aistudio.google.com/](https://aistudio.google.com/)
* ✅ 能正常访问 [https://codeassist.google/](https://codeassist.google/)
* ✅ OAuth 登录成功（日志里已经显示 `Authentication succeeded`）
* ❌ 最后被 Gemini CLI 拒绝：

  > **Your current account is not eligible for Gemini Code Assist for individuals because it is not currently available in your location.**

这说明：

**你的 Google 账号可以使用 Google AI Studio，但不能使用免费的 Gemini Code Assist 登录方式。**

这两个产品虽然都属于 Gemini，但后台权限是分开的。

---

### 推荐方案：使用 API Key

对于你的情况（中国大陆 + Clash + 个人 Google 账号），我建议直接使用：

> **Gemini API Key**

而**不要再使用 Google 登录（Code Assist）**。

原因：

* 不依赖 Code Assist 地区开放
* 登录过程简单
* Gemini CLI 官方支持
* 对个人开发者来说更稳定

---

#### 是否收费？

这是大家最容易混淆的地方。

#### Google 登录（Code Assist）

```
gemini
↓
1. Sign in with Google
```

使用的是

> Gemini Code Assist for Individuals

这是免费的，但目前**地区有限制**。

你的账号已经证明：

> **没有资格使用。**

---

#### API Key

```
gemini
↓
2. Use Gemini API Key
```

这是调用 Gemini API。

但是：

**Google AI Studio 提供了免费额度（Free Tier）**。

对于个人开发、写代码、日常使用，免费额度通常已经足够。只有超出免费额度或主动切换到付费项目时，才会产生 API 费用。免费额度和具体限制可能会调整，因此建议以 Google AI Studio 当前页面显示的配额为准。

---

#### 你的账号实际上更适合 API Key

既然：

```
https://aistudio.google.com/
```

能够正常打开，

说明你的账号已经具有：

* API Key 权限
* 免费配额（通常如此）
* Gemini API 使用资格

因此完全没有必要继续折腾 Code Assist。




---

### 为什么 Google 登录失败，而 API Key 可以？

因为它们走的是两套不同的授权体系：

| 登录方式           | 后端服务                 | 地区限制                  | 适合你 |
| -------------- | -------------------- | --------------------- | --- |
| Google Sign In | Gemini Code Assist   | 有                     | ❌   |
| Gemini API Key | Google AI Studio API | 较少（取决于 AI Studio 可用性） | ✅   |
| Vertex AI      | Google Cloud         | 企业/GCP                | 不需要 |

所以你的情况属于：

> **Code Assist 没资格，但 AI Studio 有资格。**

这是完全可能的，而且不少用户都遇到过类似情况。

---

### 结合你的使用场景

从我们之前的交流来看，你主要会用 Gemini CLI 来：

* 编写和重构 C/C++（CS144、CS:APP）
* Python 开发
* RAG / LLM 工程
* Obsidian 插件
* VS Code 辅助编码

对于这些用途，**API Key 模式已经足够，而且通常是更稳妥的选择**。

如果你愿意，我还可以告诉你如何配置 Gemini CLI，使它默认使用 API Key，不再每次启动都询问认证方式。


---

## 2：使用 Gemini API Key

如果你这样配置：

```bash
export GEMINI_API_KEY=xxxx
```

或者：

```
gemini auth api-key
```

那么 Gemini CLI 就会调用 **Gemini Developer API**。

此时：

- 免费 API Key：有免费额度（例如每天约 250 次请求，仅限 Flash 模型）
    
- 超过免费额度，或使用付费 API Key，则**按照 API 的 token 用量计费**。([Google Gemini](https://google-gemini.github.io/gemini-cli/docs/quota-and-pricing.html?utm_source=chatgpt.com "Gemini CLI: Quotas and Pricing | gemini-cli"))
    

也就是说：

```
Gemini CLI
        │
        ├── Google Login
        │      ↓
        │  免费，不计 Token
        │
        └── API Key
               ↓
        Gemini API
               ↓
         按 Token 计费（超出免费额度后）
```

### 建议步骤

### ① 创建 API Key

进入：

> [https://aistudio.google.com/](https://aistudio.google.com/)

左侧一般可以找到：

```
Get API Key
```

或者

```
Create API Key
```

创建一个新的 Key。

---

### ② 设置环境变量

Windows PowerShell：

```powershell
setx GEMINI_API_KEY "AIzaSyxxxxxxxxxxxxxxxx"
```

重新打开终端。

验证：

```powershell
echo $env:GEMINI_API_KEY
```

Git Bash：

```bash
echo $GEMINI_API_KEY
```

---

### ③ 启动 Gemini CLI

运行：

```bash
gemini
```

选择：

```
2. Use Gemini API Key
```

输入（或让它读取环境变量）即可开始使用。

---





## 3. Vertex AI   
When using Vertex AI, you must specify either:   
• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.     
• GOOGLE_API_KEY environment variable (if using express mode). 
Update your environment and try again (no reload needed if using .env)!   




可以，你已经具备了使用 **Vertex AI** 的基础条件：

* ✅ 已安装 `gcloud`
* ✅ 已登录 Google Cloud
* ✅ 已有 GCP Project：`ssim-01`
* ✅ Notebook/VM 也属于该 Project

不过 **Gemini CLI 使用 Vertex AI** 还需要几个配置。下面按顺序检查。

---

### 第一步：确认 Vertex AI API 已启用

先运行：

```bash
gcloud services list --enabled | grep aiplatform
```

如果看到：

```text
aiplatform.googleapis.com
```

说明已经启用。

如果没有：

```bash
gcloud services enable aiplatform.googleapis.com
```

---

### 第二步：确认 Project ID

你的配置显示：

```text
project = ssim-01
```

验证一下：

```bash
gcloud config get-value project
```

应该输出

```text
ssim-01
```

---

### 第三步：设置环境变量

Gemini CLI 需要两个变量：

```bash
GOOGLE_CLOUD_PROJECT
GOOGLE_CLOUD_LOCATION
```

例如：

Git Bash：

```bash
export GOOGLE_CLOUD_PROJECT=ssim-01
export GOOGLE_CLOUD_LOCATION=us-central1
```

Windows PowerShell：

```powershell
$env:GOOGLE_CLOUD_PROJECT="ssim-01"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
```

永久：

```powershell
setx GOOGLE_CLOUD_PROJECT "ssim-01"
setx GOOGLE_CLOUD_LOCATION "us-central1"
```

然后验证：

```bash
echo $GOOGLE_CLOUD_PROJECT
echo $GOOGLE_CLOUD_LOCATION
```

---

### 第四步：登录 ADC（最容易漏）

很多人已经：

```bash
gcloud auth login
```

但是 **Gemini CLI 不读取这个登录**。

它需要的是：

```bash
gcloud auth application-default login
```

执行：

```bash
gcloud auth application-default login
```

浏览器登录。

然后检查：

```bash
gcloud auth application-default print-access-token
```

如果能打印一串：

```
ya29....
```

说明 ADC 已经配置成功。

---

### 第五步：检查 Vertex AI 权限

运行：

```bash
gcloud auth list
```

应该看到：

```
ACTIVE
@gmail.com
```

然后查看：

```bash
gcloud projects get-iam-policy ssim-01
```

至少需要拥有类似角色：

* Vertex AI User
* Editor
* Owner

如果你自己创建了这个 Project，一般默认就是 Owner。

---

### 第六步：确认区域支持 Gemini

建议：

```
us-central1
```

或者：

```
us-east5
```

不要用：

```
asia-east1
```

Gemini 模型支持区域有限。

---

### 第七步：启动 Gemini CLI

运行：

```bash
gemini
```

选择：

```
3. Vertex AI
```

它会检测：

```
GOOGLE_CLOUD_PROJECT
GOOGLE_CLOUD_LOCATION
```

如果都存在，就不会再报：

```
When using Vertex AI...
```

---

### 如果仍然报错

运行下面几个命令，把输出贴给我：

```bash
gcloud config list
```

```bash
gcloud auth list
```

```bash
gcloud auth application-default print-access-token
```

```bash
echo $GOOGLE_CLOUD_PROJECT
```

```bash
echo $GOOGLE_CLOUD_LOCATION
```

以及：

```bash
gcloud services list --enabled | grep aiplatform
```

---

## 对于个人开发，我不建议优先使用 Vertex AI

结合你目前的情况：

* 可以访问 Google AI Studio
* 不是企业团队
* 没有必须依赖 GCP IAM 或企业资源

**Gemini CLI + AI Studio API Key** 往往更简单：

* 不需要配置 GCP 项目
* 不需要配置 IAM
* 不需要 ADC
* 不需要 Vertex AI API
* 通常还有免费额度，适合个人使用

Vertex AI 更适合企业环境，例如团队统一管理、服务账号、IAM 权限控制或与其他 GCP 服务集成。

所以如果你的目标只是把 Gemini CLI 用起来，我建议优先尝试 **API Key** 方案；如果你的目标是熟悉 GCP、Vertex AI 或未来部署企业级 AI 应用，再继续配置 Vertex AI 会更有价值。


# Cost

## 如何确认自己会不会收费？

运行：

```bash
gemini
```

然后输入：

```
/auth
```

如果看到类似：

```
Authenticated as:
your@gmail.com
Method: Google
```

那么就是**免费模式**。

如果看到：

```
Method: API Key
```

那么就是 API 模式，需要关注 API 配额和可能的费用。

---

## 如何查看使用量？

Gemini CLI 内置了统计命令：

```text
/stats
```

可以查看：

- 已发送请求数
    
- Token 使用情况（API Key 模式下）
    
- Cache 命中
    
- 当前额度等。([Google Gemini](https://google-gemini.github.io/gemini-cli/docs/cli/token-caching.html?utm_source=chatgpt.com "Token Caching and Cost Optimization | gemini-cli"))
    

---

## 对于像你这样的开发者（CS144、C++、Python、NLP 项目）

如果你的用途主要是：

- 阅读代码
    
- 修改 C++ 项目
    
- 调试网络协议
    
- 编写 Python
    
- 分析仓库
    

我建议**直接使用 Google 登录模式**，原因是：

- 基本不会产生费用。
    
- 每天 1000 次请求对个人开发通常已经很充裕。
    
- 无需管理 API Key 或担心意外扣费。([Google Gemini](https://google-gemini.github.io/gemini-cli/docs/quota-and-pricing.html?utm_source=chatgpt.com "Gemini CLI: Quotas and Pricing | gemini-cli"))
    

只有当你需要：

- 编写长时间自动化 Agent；
    
- 自己开发基于 Gemini API 的应用；
    
- 希望突破免费请求限制；
    

才建议切换到 API Key 或 Vertex AI 的付费模式。


