p = "/root/autodl-tmp/OpenViTac/policy/openpi/transforms.py"
s = open(p).read()
old = "    return _resize_direct_cv2(array, size, size)\n"
new = '''    # LOCAL DIAGNOSTIC (not committed): keep the native-resolution image for selected calls.
    import os as _os
    _raw = _os.environ.get("TACGUIDE_RAW")
    if _raw:
        _n = getattr(resize_rgb_image, "_n", 0); resize_rgb_image._n = _n + 1
        _per = int(_os.environ.get("TACGUIDE_RAW_PER_STEP", "2")); _step, _cam = divmod(_n, _per)
        if _step in (0, 20, 40, 60, 100):
            _os.makedirs(_raw, exist_ok=True)
            np.save(_os.path.join(_raw, f"raw_step{_step:03d}_cam{_cam}_{array.shape[0]}x{array.shape[1]}.npy"), array)
''' + old
assert s.count(old) == 1 and "TACGUIDE_RAW" not in s
open(p, "w").write(s.replace(old, new, 1)); print("patched")
