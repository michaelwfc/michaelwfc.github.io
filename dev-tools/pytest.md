# pytest in pycharm

## 配置测试框架

以pytest方式运行，需要改该工程设置默认的运行器：file->Setting->Tools->Python Integrated Tools->项目名称->Default test runner->选择py.test

## show the std

"Edit Configurations..." -> "Templates" -> "Python tests" -> "pytest"  add the -s option

## 创建测试

在编辑器中，将光标放在类声明或方法中的位置。

- 从主菜单中，选择 Navigate -> Test。
- 编辑器内，右键上下文菜单中选择 Go to -> Test (⌘⇧T: Ctrl + Shift + T)

PyCharm 显示可用测试的列表。单击"创建新测试"。在打开 Create test 对话框中进行设置， 点击 OK 会自动生成测试文件 test_rectangle 与 测试方法模板。

# pytest in vscode

# pytest.ini

```ini
[pytest]
log_cli = true
# supress the DeprecationWarning when debug pytest
filterwarnings =
     ignore::DeprecationWarning
```
