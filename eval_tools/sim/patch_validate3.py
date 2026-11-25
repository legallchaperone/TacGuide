import re
p = "/root/autodl-tmp/OpenViTac/policy/openpi/deploy_policy.py"
s = open(p).read()
a = s.index("            # LOCAL VALIDITY CHECK (not committed): reject episodes")
b = s.index("            # LOCAL DIAGNOSTIC (not committed): dump each query")
block = s[a:b]
s = s[:a] + s[b:]
block = "".join((l[4:] if l.startswith("    ") else l) for l in block.splitlines(True))
end = '                print("TACGUIDE start check: could not read tactile for vision client:", _e, flush=True)\n'
assert s.count(end) == 1
s = s.replace(end, end + block, 1)
# trace for temporal-ensemble queries as well
t_old = "            chunk = self._normalize_server_chunk(self.client.infer(obs), obs, task)\n            if self.temporal_ensemble_space"
assert s.count(t_old) == 1
t_new = '''            chunk = self._normalize_server_chunk(self.client.infer(obs), obs, task)
            import os as _os
            _trace = _os.environ.get("TACGUIDE_TRACE")
            if _trace:  # LOCAL DIAGNOSTIC (not committed)
                _os.makedirs(_trace, exist_ok=True)
                _d = {k.replace("/", "__"): np.asarray(v) for k, v in obs.items() if not isinstance(v, (str, bool))}
                _d["action_chunk"] = np.asarray(chunk)
                np.savez_compressed(_os.path.join(_trace, f"q{current_step:05d}.npz"), **_d)
            if self.temporal_ensemble_space'''
s = s.replace(t_old, t_new, 1)
open(p, "w").write(s); print("patched")
