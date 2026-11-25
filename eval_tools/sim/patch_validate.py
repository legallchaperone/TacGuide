p = "/root/autodl-tmp/OpenViTac/policy/openpi/deploy_policy.py"
s = open(p).read()
anchor = "            # LOCAL DIAGNOSTIC (not committed): dump each query's observation and action chunk.\n"
new = '''            # LOCAL VALIDITY CHECK (not committed): reject episodes whose simulator start-up state is broken.
            import os as _os0
            if _os0.environ.get("TACGUIDE_VALIDATE") and self._policy_step_index == 0:
                import cv2 as _cv2, json as _json
                _rep = {"wrist_mean": float(np.asarray(obs["observation/wrist_image"]).mean())}
                _bad = _rep["wrist_mean"] > 132.0
                _ref = _os0.environ.get("TACGUIDE_TACTILE_REF")
                _ref = np.load(_ref).astype(np.float64) if _ref and _os0.path.exists(_ref) else None
                for _side in ("left", "right"):
                    _k = f"observation/{_side}_tactile_clip"
                    if _k not in obs:
                        continue
                    _im = np.asarray(obs[_k])[-1]
                    _g = _cv2.cvtColor(_im, _cv2.COLOR_RGB2GRAY)
                    _n, _lab, _st, _ = _cv2.connectedComponentsWithStats((_g < 45).astype(np.uint8), connectivity=8)
                    _cnt = int(sum(1 for _i in range(1, _n) if 6 <= _st[_i, _cv2.CC_STAT_AREA] <= 400))
                    _frac = float((_g < 45).mean())
                    _rep[f"{_side}_markers"] = _cnt; _rep[f"{_side}_dark_frac"] = _frac
                    if _ref is not None and _ref.shape == _im.shape:
                        _rep[f"{_side}_ref_mad"] = float(np.abs(_im.astype(np.float64) - _ref).mean())
                    _bad = _bad or _cnt != 63 or not (0.070 <= _frac <= 0.082)
                _rep["bad_start"] = bool(_bad)
                _vd = _os0.environ.get("TACGUIDE_TRACE", "/tmp")
                _os0.makedirs(_vd, exist_ok=True)
                open(_os0.path.join(_vd, "START_CHECK.json"), "w").write(_json.dumps(_rep))
                print("TACGUIDE start check:", _rep, flush=True)
                if _bad:
                    raise RuntimeError(f"tacguide_bad_start: {_rep}")
'''
assert anchor in s and "TACGUIDE_VALIDATE" not in s
open(p, "w").write(s.replace(anchor, new + anchor, 1)); print("patched")
