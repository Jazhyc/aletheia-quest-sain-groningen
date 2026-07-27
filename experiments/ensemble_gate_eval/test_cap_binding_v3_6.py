"""Does MAX_CAP bind at real Iris intensity? Sweep the cap against both regimes.

The earlier sweep in ``test_saturation_v3_6.py`` found no cap effect, but it
reached only a +0.025 probe edge because sharpening the probe saturates. This
script creates the edge by *blunting the judge* instead, which does not
saturate, and so reaches the real Iris condition (+0.068).

Iris side: judge blunted to ``probe_auroc - edge``.
Notus side: probe blunted to the measured 0.5586 (and 0.48 inverted), judge left
healthy. The sign gate is kept in every arm -- only MAX_CAP moves.
"""
import sys; sys.argv=['x']
src=open('/home/matthijs/programming/aletheia-quest-sain-groningen/experiments/ensemble_gate_eval/test_saturation_v3_6.py').read()
exec(src.split('VARIANTS = [')[0])
from pathlib import Path as _P
REPO_ROOT=_P("/home/matthijs/programming/aletheia-quest-sain-groningen")
ACTIVATIONS=REPO_ROOT/"results/whitebox/activations"
JUDGE_CACHE=REPO_ROOT/"experiments/ensemble_gate_eval/judge_cache.json"

rng=np.random.default_rng(SEED)
jc={tuple(k.split("|",1)):(float(v[0]),float(v[1])) for k,v in json.loads(JUDGE_CACHE.read_text()).items()}
data=load_datasets(jc)

folds=[]
for base in sorted({d["base"] for d in data}):
    g=[d for d in data if d["base"]==base]
    if len(g)<3: continue
    for test in g:
        tr=[d for d in g if d["name"]!=test["name"]]; y=test["y"]
        if len(np.unique(y))<2: continue
        xt=np.concatenate([d["x"] for d in tr]); yt=np.concatenate([d["y"] for d in tr])
        mu,sd=xt.mean(0),xt.std(0)+1e-6
        m=LogisticRegression(max_iter=2000,C=0.01,random_state=SEED).fit((xt-mu)/sd,yt)
        pz=zscore(m.decision_function((test["x"]-mu)/sd)); pa=roc_auc_score(y,pz)
        folds.append(dict(pz=pz,y=y,pa=pa,judge=test["judge"]))

def gate(jz,pz,max_cap):
    ag=(jz*pz>0).astype(float); cap=BASE_CAP+ag*(max_cap-BASE_CAP)
    return jz+cap*np.tanh(pz)

print(f"{'target edge':>12s}{'judge':>9s}{'probe':>9s}" + "".join(f"{f'cap{c}':>9s}" for c in (2,4,6,8,12,20,40)) + f"{'best':>9s}")
for edge in (0.025,0.045,0.068,0.090):
    per={c:[] for c in (2,4,6,8,12,20,40)}; ja=[];pa=[]
    for f in folds:
        tgt=f["pa"]-edge
        if tgt<0.51: continue
        jz=retune_auroc(f["judge"],f["y"],tgt,rng,sharpen_ok=False)
        ja.append(roc_auc_score(f["y"],jz)); pa.append(f["pa"])
        for c in per: per[c].append(roc_auc_score(f["y"],gate(jz,f["pz"],c*STEP)))
    mj,mp=np.mean(ja),np.mean(pa)
    cells={c:np.mean(v) for c,v in per.items()}
    best=max(cells,key=cells.get)
    print(f"{edge:>12.3f}{mj:>9.4f}{mp:>9.4f}"+"".join(f"{cells[c]:>9.4f}" for c in (2,4,6,8,12,20,40))+f"{'cap'+str(best):>9s}")
    caps={c:(cells[c]-mj)/(mp-mj) for c in cells}
    print(f"{'   capture%':>12s}{'':9s}{'':9s}"+"".join(f"{caps[c]*100:>8.0f}%" for c in (2,4,6,8,12,20,40)))

print("\n=== NOTUS side: probe blunted to measured 0.5586, judge healthy ===")
print(f"{'probe q':>9s}" + "".join(f"{f'cap{c}':>9s}" for c in (2,4,6,8,12,20,40)))
notus={}
for q in (0.5586,0.48):
    per={c:[] for c in (2,4,6,8,12,20,40)}
    for f in folds:
        jz=zscore(f["judge"])
        bp=retune_auroc(f["pz"],f["y"],q,rng,sharpen_ok=False)
        if q<0.5: bp=-bp
        for c in per: per[c].append(roc_auc_score(f["y"],gate(jz,bp,c*STEP)))
    notus[q]={c:np.mean(v) for c,v in per.items()}
    print(f"{q:>9.4f}"+"".join(f"{notus[q][c]:>9.4f}" for c in (2,4,6,8,12,20,40)))

print("\n=== NET: headline delta vs cap4, using real-Iris edge 0.068 + Notus 0.5586 ===")
iris={}
per={c:[] for c in (2,4,6,8,12,20,40)}; ja=[];pa=[]
for f in folds:
    tgt=f["pa"]-0.068
    if tgt<0.51: continue
    jz=retune_auroc(f["judge"],f["y"],tgt,rng,sharpen_ok=False)
    ja.append(roc_auc_score(f["y"],jz)); pa.append(f["pa"])
    for c in per: per[c].append(roc_auc_score(f["y"],gate(jz,f["pz"],c*STEP)))
iris={c:np.mean(v) for c,v in per.items()}
print(f"{'cap':>6s}{'d_iris':>10s}{'d_notus':>10s}{'d_head':>10s}")
for c in (2,4,6,8,12,20,40):
    di=iris[c]-iris[4]; dn=notus[0.5586][c]-notus[0.5586][4]
    print(f"{c:>6d}{di:>+10.4f}{dn:>+10.4f}{(di+dn)/2:>+10.4f}")
