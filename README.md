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
#### 内有中文注释,训练迭代数可自行调大
如遇到以下报错</br>
```
cannot import name '_validate_lengths' from 'numpy.lib.arraypad'
```
原因：</br>
这是在解决skimage0.15版本后出现的问题。找不到_validate_lengths函数，在arraypad.py文件中确实找不到对应的函数，所以找到以前配置过的环境中对应的文件，拷贝这个缺失的函数。<br/>
python3.7/site-packages/numpy/lib/arraypad.py,打开文件后，在954后添加以下代码，保存退出，问题解决。<br/>
```

def _normalize_shape(ndarray, shape, cast_to_int=True):
    """
    Private function which does some checks and normalizes the possibly
    much simpler representations of ‘pad_width‘, ‘stat_length‘,
    ‘constant_values‘, ‘end_values‘.

    Parameters
    ----------
    narray : ndarray
        Input ndarray
    shape : {sequence, array_like, float, int}, optional
        The width of padding (pad_width), the number of elements on the
        edge of the narray used for statistics (stat_length), the constant
        value(s) to use when filling padded regions (constant_values), or the
        endpoint target(s) for linear ramps (end_values).
        ((before_1, after_1), ... (before_N, after_N)) unique number of
        elements for each axis where `N` is rank of `narray`.
        ((before, after),) yields same before and after constants for each
        axis.
        (constant,) or val is a shortcut for before = after = constant for
        all axes.
    cast_to_int : bool, optional
        Controls if values in ``shape`` will be rounded and cast to int
        before being returned.

    Returns
    -------
    normalized_shape : tuple of tuples
        val                               => ((val, val), (val, val), ...)
        [[val1, val2], [val3, val4], ...] => ((val1, val2), (val3, val4), ...)
        ((val1, val2), (val3, val4), ...) => no change
        [[val1, val2], ]                  => ((val1, val2), (val1, val2), ...)
        ((val1, val2), )                  => ((val1, val2), (val1, val2), ...)
        [[val ,     ], ]                  => ((val, val), (val, val), ...)
        ((val ,     ), )                  => ((val, val), (val, val), ...)

    """
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
    """
    Private function which does some checks and reformats pad_width and
    stat_length using _normalize_shape.

    Parameters
    ----------
    narray : ndarray
        Input ndarray
    number_elements : {sequence, int}, optional
        The width of padding (pad_width) or the number of elements on the edge
        of the narray used for statistics (stat_length).
        ((before_1, after_1), ... (before_N, after_N)) unique number of
        elements for each axis.
        ((before, after),) yields same before and after constants for each
        axis.
        (constant,) or int is a shortcut for before = after = constant for all
        axes.

    Returns
    -------
    _validate_lengths : tuple of tuples
        int                               => ((int, int), (int, int), ...)
        [[int1, int2], [int3, int4], ...] => ((int1, int2), (int3, int4), ...)
        ((int1, int2), (int3, int4), ...) => no change
        [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...)
        ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)
        [[int ,     ], ]                  => ((int, int), (int, int), ...)
        ((int ,     ), )                  => ((int, int), (int, int), ...)

    """
    normshp = _normalize_shape(narray, number_elements)
    for i in normshp:
        chk = [1 if x is None else x for x in i]
        chk = [1 if x >= 0 else -1 for x in chk]
        if (chk[0] < 0) or (chk[1] < 0):
            fmt = "%s cannot contain negative values."
            raise ValueError(fmt % (number_elements,))
    return normshp
    

```


