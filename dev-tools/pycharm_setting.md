# 自动生成

## 文件头部注释
setting > Editor > File and Code Template > Python Script

```python
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#File    :   ${NAME}.py
#Desc:
#Time: ${DATE} ${TIME}
#Contact :  
#License :   Create by $USER on $DATE, Copyright $YEAR $USER. All rights reserved.

#Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
${DATE} ${TIME}   $USER      1.0         
```

## 函数注释
File-》Setting-》Editor-》Live Templates-》Python进入代码片段编辑界面

```python
def $NAME$($args$): 
    """
    """  
    #time: $date$ $time$   
    #author: $user$ 
    #version: 1.0  
    #description:        
    #params: $params$ 
    #return: $returns$ 
    """ 
    $END$ 
    #TODO 
    pass
```