
# Extensions

 .vscode/extensions.json

```json
{
    "recommendations": [
        "Python",
        "autoDocstring: VSCode Python Docstring Generator",
        "Flake8",
        "GitLens — Git supercharged",
        "Tabnine: AI Autocomplete & Chat for ",
        "ESLint",
        "Prettier - Code formatter"
    ]
}
```

## Python extensions

|Python||
|----|-----|
|autoDocstring: VSCode Python Docstring Generator| 函数和方法的注释文档docstring，其重要性不需要再强调了，安装了autoDocstring插件后，通过快捷键：“ctrl+shift+2”，可以自动生成Google或Numpy风格的注释文档。autoDocstring的配置方法：从“File”菜单->Preferences->Settings，打开“Settings”界面，在搜索栏中键入：“autoDocstring”，根据自己的风格喜好|
|Flake8,pep8,pylint| 如果要使用 flake8 或要想 flake8 等工具起作用，前提是必须把 settings.json 文件中的"python.linting.enabled"值设为“true”|
|yapf、black,autopep8| Python 自动格式化代码通常用 |
|koroFileHeader |自动添加 头部注释 和 函数注释 的插件。支持自定义内容，需要在 settings.json 中进行自定义配置|
|GitLens — Git supercharged| |
|Better Comments|这个插件通过不同的彩色把不同功能的注释信息区分开来。能区分的功能有：Alerts、Queries、TODOs、Highlights和用户自定义，如下图所示。  Better Comments的使用方法：Better Comments安装好后，"!"表示警告、"?"表示询问，"TODOs"表示待办事项，"*"表示高亮内容|

## AI code extensions

|AI code extensions||
|--------|----|
|Tabnine: AI Autocomplete & Chat for ||

## Javacript extensions

|Javacript extensions||
|--------|----|
|ESLint||
|Prettier - Code formatter||

# settings

.vscode/settings.json

```json
{
   "files.autoSaveWhenNoErrors": true,

  "python.defaultInterpreterPath":"C:\\Users\\***\\Anaconda\\envs\\sec_master\\python.exe",
   "python.envFile": "${workspaceFolder}/dev.env",

  "terminal.integrated.defaultProfile.windows": "Git Bash",
  "terminal.integrated.cwd": "${workspaceFolder}",
  "terminal.integrated.fontSize":10,
  /* 
  To configure Python to search for modules in the src-folder we alter the default search path. In PyCharm this is done by selecting a source folder. In Visual Studio Code, this is done by setting the PYTHONPATH variable.  
  */
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "${env:PYTHONPATH};${workspaceFolder}${pathSeparator}src"
  },
  "terminal.integrated.automationProfile.windows": { 
        "GitBash":{
            "path": "C:\\Program Files\\Git\\bin\\bash.exe" 
        } 
   }, 
  "terminal.integrated.defaultProfile.windows": "Git Bash",
  "terminal.integrated.fontSize": 10,

   //自动检查代码 
    "python.linting.enabled": true, 
    "python.linting.pylintEnabled": false, 
    "python.linting.pylintArgs": [ 
        "--disable=wrong-import-order,wrong-import-position,unused-import,ungrouped-imports,line-too-long,logging-fstring-interpolation" 
    ], 
    "python.linting.flake8Enabled": true,


  // 单元测试 
    "python.testing.cwd": "${workspaceFolder}${pathSeparator}src",

    "python.testing.autoTestDiscoverOnSaveEnabled": true, 
    "python.testing.pytestEnabled": true, 
    "python.testing.pytestPath": "C:\\Users\\*****\\Anaconda\\envs\\sec_master\\Scripts\\pytest.exe", 
    "python.testing.pytestArgs": [ 
        "--no-cov", 
        "-s" 
    ],

  //Indicates whether to automatically add search paths based on some predefined names (like src)
  "python.analysis.autoSearchPaths": true,
    //Specifies extra search paths for import resolution.
  "python.analysis.extraPaths": [
    "${workspaceFolder}${pathSeparator}Langchain-Chatchat"
  ],

  "markdown-preview-github-styles.colorTheme": "light",
  "autoDocstring.docstringFormat": "google-notypes",

  "workbench.settings.editor": "json",
  //主题颜色  
  "workbench.colorTheme": "Monokai", // "Visual Studio Dark",

  "editor.fontSize": 10,
  "window.zoomLevel": 2,
  "editor.wordWrap": "on",
  "editor.detectIndentation": false,

  "files.associations": { 
  "*.vue": "vue", 
  "*.wpy": "vue", 
  "*.wxml": "html", 
  "*.wxss": "css" 
  },
  // 重新设定tabsize
  "editor.tabSize": 2,
  //失去焦点后自动保存
  "files.autoSave": "onFocusChange",
  // #值设置为true时，每次保存的时候自动格式化；
  "editor.formatOnSave": false,
   //每120行就显示一条线
  "editor.rulers": [
  ],
  // 在使用搜索功能时，将这些文件夹/文件排除在外
  "search.exclude": {
      "**/node_modules": true,
      "**/bower_components": true,
      "**/target": true,
      "**/logs": true,
  }, 
  // 这些文件将不会显示在工作空间中
  "files.exclude": {
      "**/.git": true,
      "**/.svn": true,
      "**/.hg": true,
      "**/CVS": true,
      "**/.DS_Store": true,
      "**/Thumbs.db": true,
      "**/*.js": {
          "when": "$(basename).ts" //ts编译后生成的js文件将不会显示在工作空中
      },
      "**/node_modules": true
  }
}

```

# Debugging

.vscode/languch.json

```json
{ 
    /* 
      Note that the PYTHONPATH must be set for both the editors’ Python environment and the integrated terminal. 
The editors’ Python environment is used by extensions and provides linting and testing functionality. 
The integrated terminal is used when debugging to activate a new python environment.
 */
    "configurations": [ 
        { 
            "name": "Python Debugger:Current File", 
            "type": "python", 
            //request: "launch" or "attach": start the debugger on the file specified in program 
            "request": "launch", 
            // 
            "program": "${file}", 
            // when debugging tests in VS Code,important to setting when do unit test 
            "purpose": [ 
                "debug-test" 
            ], 
            //Specifies how program output is displayed  
            "console": "integratedTerminal", 
            "justMyCode": false, 
            //When set to true, breaks the debugger at the first line of the program being debugged.  
            "stopOnEntry":true, 
            // with the arguments --port 1593 when you start the debugger 
            //"args" : ["--port", "1593"] 
        }, 
        { 
            "name": "Python: main entry point", 
            "type": "python", 
            "request": "launch", 
            "program": "${workspaceFolder}/src/main.py", 
        },
        {
            "name": "node debugger current file",
            "type": "node",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "NODE_OPTIONS": "--openssl-legacy-provider",
                // "HTTP_PROXY": "http://127.0.0.1:7890",
                // "HTTPS_PROXY": "http://127.0.0.1:7890"
            }
        },
        {
            "name": "Lannch Chrome against localhost",
            "type": "chrome",
            "request": "launch",
            "url": "http://localhost:3000",
            "webRoot": "${workspaceFolder}${pathSeparator}spotify-web-playback"
        }


    ] 
}
```

# Unit test

<https://code.visualstudio.com/docs/python/testing#_intellisense-for-pytest>
<https://www.youtube.com/watch?v=ucjRpS7WCPA>
<https://blog.thenets.org/how-to-create-a-modern-pytest-dev-environment-with-vscode/>

# Shotcuts

|command | function |
|--------|----------|
|Ctrl + `| show terminal|
|Shift + Alt + F|格式化代码|
|Shift + Alt +A |块区域注释|
|Ctrl+Shift+N |打开新的编辑器窗口|

## add shortcuts

1. Press CTRL+SHIFT+P.
2. Type:  Open keyboard shortcuts (json)
3. Select An editor will appear with keybindings.json file. Place the following JSON in there and save:

```json
[
    {
        "key": "ctrl+shift+u",
        "command": "editor.action.transformToUppercase",
        "when": "editorTextFocus"
    },
    {
        "key": "ctrl+shift+l",
        "command": "editor.action.transformToLowercase",
        "when": "editorTextFocus"
    },
    { 
        // Navigate to Top of Call Stack
        "key":"alt+f10", 
        "command":"workbench.action.debug.callStackTop" 
    }
]
```

# code templates

1. In vscode, File -> Preferences -> Configure User Snippets. Type python and choose python. A json file will open
2. Copy-paste all or the specific snippets you want in the file and save
3. Ctrl+Shift+P then Reload Window to activate the changes

This is the default main snippet:

## [vs-code-snippets](https://adamtheautomator.com/vs-code-snippets/#Creating_a_Custom_Python_Snippet)

[snippet-syntax](https://code.visualstudio.com/docs/editor/userdefinedsnippets#_snippet-syntax)

```json
"if(main)": {
        "prefix": "main",
        "body": ["if __name__ == \"__main__\":", "    ${1:pass}"],
        "description": "Code snippet for a `if __name__ == \"__main__\": ...` block"
    },
```

If you want to change or tweak which text triggers the snippet, modify the prefix field. The prefix field can be a string as shown above or a list if you want more triggers:

```json
"prefix": ["__main__", "ifmain", "main", "ifm", "if m"],
```
