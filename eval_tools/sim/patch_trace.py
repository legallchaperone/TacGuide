p = "/root/autodl-tmp/OpenViTac/policy/openpi/deploy_policy.py"
s = open(p).read()
old = '''            self._chunk_step = 0
            print(f"OpenPI action chunk: {self._action_chunk.shape}")
'''
new = old + '''            # LOCAL DIAGNOSTIC (not committed): dump each query's observation and action chunk.
            import os as _os
            _trace = _os.environ.get("TACGUIDE_TRACE")
            if _trace:
                _os.makedirs(_trace, exist_ok=True)
                _d = {k.replace("/", "__"): np.asarray(v) for k, v in obs.items() if not isinstance(v, (str, bool))}
                _d["action_chunk"] = np.asarray(self._action_chunk)
                np.savez_compressed(_os.path.join(_trace, f"q{self._policy_step_index:05d}.npz"), **_d)
'''
assert old in s and "TACGUIDE_TRACE" not in s
open(p, "w").write(s.replace(old, new, 1))
print("patched")
