# Kit verify, part 2: runs the target graph cold / warm / exchange off / after a release, checks the module and writes VERIFY_REPORT.md.
# Thresholds can be overridden by the builder (KIT_LIMITS dict).
import ast
import copy

LIM = dict(rel_err=3e-2, margin_gib=1.0, exchange_share=0.15, release_gib=3.0)
LIM.update(globals().get('KIT_LIMITS', {}))
g0 = copy.deepcopy(KIT_GRAPH)
SAMPLER_IDS = {nid for nid, n in g0.items() if n['class_type'] in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced')}
for nid, n in list(g0.items()):
    if n['class_type'] in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced'):
        g0['zz_fc_' + nid] = {'class_type': 'FiniteCheck', 'inputs': {'latent': [nid, 0]}}


def seeded(g, seed):
    g = copy.deepcopy(g)
    for n in g.values():
        for k in ('noise_seed', 'seed'):
            if isinstance(n['inputs'].get(k), int):
                n['inputs'][k] = seed
    return g


def step_stats(t0, t1):
    ticks = {}
    for t, m in EVENTS:
        if t0 <= t <= t1 and m.get('type') == 'progress':
            d = m['data']; ticks.setdefault((d.get('node'), d.get('max')), []).append((t, d.get('value')))
    if not ticks:
        return None
    own = [k for k in ticks if k[0] in SAMPLER_IDS]
    if len(own) != 1:   # missing or ambiguous sampler stream is not timing evidence
        return None
    key = own[0]
    ts = [t for t, _ in ticks[key]]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    return {'steps': key[1], 'gaps_s': [round(x, 1) for x in gaps], 'median_s': round(sum(gaps) / len(gaps), 2) if gaps else None}


def peak(t0, t1):
    rows = [s for s in SAMPLES if t0 <= s[0] <= t1]
    return [round(max(r[4][i][0] for r in rows) / 1024, 2) for i in (0, 1)] if rows else None


RESULTS = {}
SWITCH_RESULTS = {}


def job(name, graph):
    t0 = time.time(); ok = run(name, graph); t1 = time.time()
    RESULTS[name] = {'ok': ok, 'seconds': round(t1 - t0, 1), 'steps': step_stats(t0, t1), 'peak_gib': peak(t0, t1)}
    say(name, json.dumps(RESULTS[name]))
    return ok


def switch(off):
    SWITCH_RESULTS[off] = run('switch_off=%s' % off, {'sw': {'class_type': 'TPSwitch', 'inputs': {'exchange_off': off}}})


job('V1_cold', seeded(g0, 1))
job('V2_warm_on', seeded(g0, 2))
switch(True)
job('V3_warm_exchange_off', seeded(g0, 3))
switch(False)
job('V4_warm_on_again', seeded(g0, 4))
run('V5_release', {'p': {'class_type': 'TPProbe2', 'inputs': {'release': True}}})
job('V6_after_release', seeded(g0, 5))

text = LOG.read_text(errors='replace')
checks = re.findall(r'\[TP-CHECK\] (\S+) rel_err=(\S+) drift=(\S+) finite=(\S+)', text)
nonfinite = [int(x) for x in re.findall(r'\[FiniteCheck\] non-finite values: (\d+)', text)]
probe = json.loads(open('/kaggle/working/h3_up/tpprobe.json').read()) if os.path.exists('/kaggle/working/h3_up/tpprobe.json') else {}
on = [RESULTS[k]['steps']['median_s'] for k in ('V2_warm_on', 'V4_warm_on_again') if RESULTS.get(k, {}).get('steps') and RESULTS[k]['steps']['median_s']]
off = RESULTS.get('V3_warm_exchange_off', {}).get('steps')
crit = []


def c(name, ok, detail):
    crit.append((name, 'PASS' if ok else ('FAIL' if ok is False else 'UNKNOWN'), detail))


c('all jobs ran', all(r['ok'] for r in RESULTS.values()), {k: r['ok'] for k, r in RESULTS.items()})
c('block check rel_err', (max(float(x[1]) for x in checks) <= LIM['rel_err']) if checks else None, checks or 'no [TP-CHECK] line in the log (the module must log it)')
c('finite outputs', (not any(nonfinite) and all(x[3] == 'True' for x in checks)) if nonfinite else None, {'FiniteCheck': nonfinite})
pk = [RESULTS[k]['peak_gib'] for k in RESULTS if RESULTS[k].get('peak_gib')]
c('memory margin (peak <= 14.56 - %.1f GiB)' % LIM['margin_gib'], (max(max(p) for p in pk) <= 14.56 - LIM['margin_gib']) if pk else None, {k: r['peak_gib'] for k, r in RESULTS.items()})
switch_logs = re.findall(r'\[TPSwitch\] exchange_off = (True|False) modules (\[[^\n]*\])', text)
switched = {}
for state, names in switch_logs:
    try:
        switched[state] = ast.literal_eval(names)
    except (ValueError, SyntaxError):
        pass
switch_ok = (all(SWITCH_RESULTS.get(state) is True for state in (True, False))
             and bool(switched.get('True')) and switched.get('True') == switched.get('False'))
c('exchange switch reached modules', switch_ok, {'jobs': SWITCH_RESULTS, 'modules': switched})
if switch_ok and len(on) == 2 and off and off.get('median_s') and min(on + [off['median_s']]) >= 1.0:   # sub-second gaps = wrong progress stream, do not trust
    share = (sum(on) / len(on) - off['median_s']) / (sum(on) / len(on))
    c('exchange share of a step <= %d%%' % int(LIM['exchange_share'] * 100), share <= LIM['exchange_share'], 'on %.1f s, off %.1f s, share %.1f%%' % (sum(on) / len(on), off['median_s'], share * 100))
else:
    c('exchange share', None, 'no exchange-off timing (module has no XCHG_OFF attribute or no step ticks)')
after = (probe.get('after') or {})
c('release frees memory (< %.1f GiB allocated per GPU)' % LIM['release_gib'], (max(v['allocated_gib'] for v in after.values()) < LIM['release_gib']) if after else None, probe)
c('reload after release works', RESULTS.get('V6_after_release', {}).get('ok'), RESULTS.get('V6_after_release'))
verdict = 'PASS' if all(s == 'PASS' for _, s, _ in crit) else ('FAIL' if any(s == 'FAIL' for _, s, _ in crit) else 'INCOMPLETE')
rep = ['# VERIFY_REPORT: %s' % verdict, ''] + ['- **%s**: %s: %s' % (n, s, json.dumps(d, default=str)[:700]) for n, s, d in crit]
rep += ['', '## Jobs', '```json', json.dumps(RESULTS, indent=1), '```', '', '## Log lines', '```'] + [l for l in text.splitlines() if re.search(r'TP-CHECK|\[MultiGPU TP|FiniteCheck|TPProbe2|TPSwitch|out of memory|Error', l)][:60] + ['```']
(OUT / 'VERIFY_REPORT.md').write_text('\n'.join(rep), encoding='utf-8')
say('\n'.join(rep[:12]))
shutil.copy(LOG, OUT / 'comfyui.log')
say('total kernel time %.0f s' % (time.time() - T0))
STOP.set()
proc.terminate()
