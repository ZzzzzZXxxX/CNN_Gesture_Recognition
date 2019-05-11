## 使用卷积神经网络CNN对手势进行识别
## step 1
```
python train.py
```
## step 2
```
python test.py
```

</br>
</br>
#### 内有中文注释,训练迭代数可自行调大 <br/>
如遇到以下报错</br>
```
cannot import name '_validate_lengths' from 'numpy.lib.arraypad'
```
原因：</br>
这是在解决skimage0.15版本后出现的问题。找不到_validate_lengths函数，在arraypad.py文件中确实找不到对应的函数，所以找到以前配置过的环境中对应的文件，拷贝这个缺失的函数。<br/>
python3.7/site-packages/numpy/lib/arraypad.py,打开文件后，在954后添加以下代码，保存退出，问题解决。<br/>


def _normalize_shape(ndarray, shape, cast_to_int=True):
    
    ndims = ndarray.ndim

    # Shortcut shape=None
    if shape is None:
        return ((None, None), ) * ndims

    # Convert any input `info` to a NumPy array
    shape_arr = np.asarray(shape)

    try:
        shape_arr = np.broadcast_to(shape_arr, (ndims, 2))
    except ValueError:
        fmt = "Unable to create correctly shaped tuple from %s"
        raise ValueError(fmt % (shape,))

    # Cast if necessary
    if cast_to_int is True:
        shape_arr = np.round(shape_arr).astype(int)

    # Convert list of lists to tuple of tuples
    return tuple(tuple(axis) for axis in shape_arr.tolist())


def _validate_lengths(narray, number_elements):
    
    normshp = _normalize_shape(narray, number_elements)
    for i in normshp:
        chk = [1 if x is None else x for x in i]
        chk = [1 if x >= 0 else -1 for x in chk]
        if (chk[0] < 0) or (chk[1] < 0):
            fmt = "%s cannot contain negative values."
            raise ValueError(fmt % (number_elements,))
    return normshp
    




