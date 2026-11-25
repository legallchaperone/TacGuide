p = "/root/autodl-tmp/OpenViTac/policy/openpi/deploy_policy.py"
s = open(p).read()
a1 = "            eef_state_mode=self.eef_state_mode,\n        )\n"
assert s.count(a1) == 1 and "_tac_check" not in s
s = s.replace(a1, a1 + '''        # LOCAL VALIDITY CHECK (not committed): for the vision-only client, read the tactile images anyway at step 0.
        import os as _osv
        self._tac_check = None
        if _osv.environ.get("TACGUIDE_VALIDATE") and self._policy_step_index == 0 and not self.send_tactile:
            try:
                _o2 = openpi_obs_from_openvitac(
                    observation=observation, prompt=self._get_prompt(task), image_size=self.image_size,
                    send_tactile=True, tactile_image_keys=self.tactile_image_keys,
                    image_color_order=self.image_color_order, control_mode=self.control_mode,
                    eef_state_mode=self.eef_state_mode,
                )
                self._tac_check = {sd: np.asarray(_o2[f"observation/{sd}_tactile_image"]) for sd in ("left", "right")}
            except Exception as _e:
                print("TACGUIDE start check: could not read tactile for vision client:", _e, flush=True)
''', 1)
b1 = '''                    if _k not in obs:
                        continue
                    _im = np.asarray(obs[_k])[-1]
'''
assert b1 in s
s = s.replace(b1, '''                    if _k in obs:
                        _im = np.asarray(obs[_k])[-1]
                    elif getattr(self, "_tac_check", None) is not None:
                        _im = self._tac_check[_side]
                    else:
                        continue
                    _refp = (_os0.environ.get("TACGUIDE_TACTILE_REF") or "").replace("_left_", f"_{_side}_")
                    _ref = np.load(_refp).astype(np.float64) if _refp and _os0.path.exists(_refp) else None
''', 1)
open(p, "w").write(s); print("patched")
