"""Rebuild two paper panels from the frozen Stage 21 search; never fit a model."""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(r"__ESCC_PROJECT_ROOT__")
STAGE = ROOT / "21_surgical_RNA_LNM_development/clinical_LNM_v2"
PACKAGE = ROOT / "SCI论文最终组图素材"
OUT = PACKAGE / "原始数据索引/RNA_feature_optimization"
STYLE = ROOT / "23_manuscript_figures/figure2/scripts"
NAMES = ("RNA_feature_number_optimization", "RNA_feature_selection_heatmap")
COUNTS = [5, 10, 20, 30, 50, 100]
CONFIG = {"C": 1.0, "penalty": "l2"}
SELECTOR = "PLS-DA"

sys.path.insert(0, str(STYLE))
from style import BLUE, INK, MUTED, ORANGE, MM, Frame, plt, unit_axis
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_csv(df, name):
    df.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.17g")


def extract():
    lock_path = STAGE / "config/final_model_lock.json"
    lock = read(lock_path)
    d = Path(lock["winner_directory"])
    model = d / "model.joblib"
    model_sha = sha(model)
    assert model_sha == lock["model_SHA256"]
    assert lock["winner_classifier"] == "Logistic Regression"
    assert lock["winner"]["recipe"] == {"selector": SELECTOR, "count": 100, "config": CONFIG}
    attempts_path = d / "all_attempts.json"
    matched_path = d / "matched_count_curve.json"
    published_curve = STAGE / "03_internal_evaluation/final_model_matched_count_curve.csv"
    full_log = STAGE / "03_internal_evaluation/all_hyperparameter_search_results.csv"
    manifest_path = STAGE / "00_inventory/surgical_patient_manifest.csv"
    source = [lock_path, model, attempts_path, matched_path, published_curve, full_log, manifest_path,
              d / "chosen.json", STAGE / "scripts/train_all_12.py", STAGE / "scripts/model_space.py",
              STYLE / "style.py"]
    attempts, matched, chosen = map(read, (attempts_path, matched_path, d / "chosen.json"))
    assert chosen == lock["winner"]
    stages = Counter(a["stage"] for a in attempts)
    assert stages == {"A_complete_factorial": 162, "B_all_classifiers_refined": 24, "C_matched_count_comparison": 6}
    assert all(a["status"] == "valid" and a["classifier"] == "Logistic Regression" for a in attempts)
    assert len({a["candidate_id"] for a in attempts}) == 186
    overall = pd.read_csv(full_log, float_precision="round_trip")
    assert len(overall) == 13824 and len(overall.query("cohort == 'full_development'")) == 2304
    lr = overall.query("cohort == 'full_development' and classifier == 'Logistic Regression'").sort_values("attempt")
    assert np.array_equal(lr.candidate_id, [a["candidate_id"] for a in attempts])
    assert np.array_equal(lr.inner_mean_AUROC, [a["inner_mean_AUROC"] for a in attempts])
    assert np.array_equal(lr.inner_SD_AUROC, [a["inner_SD_AUROC"] for a in attempts])
    assert all(r["chosen"]["candidate_id"] in set(overall.loc[
        overall.cohort.eq("full_development") & overall.classifier.eq(r["classifier"]), "candidate_id"])
        for r in lock["ranking"])
    selected_id = chosen["candidate_id"]
    winner_score = chosen["inner_mean_AUROC"]
    assert winner_score == lr.inner_mean_AUROC.max() == overall.loc[
        overall.cohort.eq("full_development") & overall.status.eq("valid"), "inner_mean_AUROC"].max()
    assert set(lr.loc[lr.inner_mean_AUROC.eq(winner_score), "candidate_id"]) == {selected_id}
    assert set(lr.loc[lr.candidate_id.eq(selected_id), "attempt"]) == {126, 192}
    a = lr[lr.stage.eq("A_complete_factorial")].copy()
    selectors = a.selector.drop_duplicates().tolist()
    assert len(selectors) == 9 and set(a.RNA_count) == set(COUNTS)
    assert len(a.groupby(["selector","RNA_count"])) == 54
    assert all({json.loads(p)["C"] for p in q.parameters} == {.01,.1,1.0}
               and all(json.loads(p)["penalty"]=="l2" for p in q.parameters)
               and q.candidate_id.nunique()==3 for _,q in a.groupby(["selector","RNA_count"]))
    heat = a[a.parameters.map(json.loads).map(lambda p: p == CONFIG)].copy()
    assert len(heat) == 54 and not heat.duplicated(["selector","RNA_count"]).any()
    assert heat.loc[heat.inner_mean_AUROC.idxmax(), "candidate_id"] == selected_id
    assert [a["recipe"]["count"] for a in matched] == COUNTS
    match_attempts = lr[lr.stage.eq("C_matched_count_comparison")].set_index("RNA_count")
    assert all(a["recipe"]["selector"] == SELECTOR and a["recipe"]["config"] == CONFIG
               and match_attempts.loc[a["recipe"]["count"],"candidate_id"] == a["candidate_id"] for a in matched)
    assert set(heat[heat.selector.eq(SELECTOR)].candidate_id) == {a["candidate_id"] for a in matched}
    manifest = pd.read_csv(manifest_path,dtype={"Patient_ID":str}).set_index("Patient_ID")
    assert manifest.index.is_unique and len(manifest) == 160 and manifest.LNM.sum() == 100
    shared_partitions = None
    rows = []
    genes = {}
    for row in heat.itertuples():
        trial_path = d / "trials" / (row.candidate_id + ".json.gz")
        source.append(trial_path)
        with gzip.open(trial_path,"rt",encoding="utf-8") as f:
            trial = json.load(f)
        assert trial["recipe"] == {"selector":row.selector,"count":row.RNA_count,"config":CONFIG}
        folds, ev = trial["fold_metrics"], trial["inner_evidence"]
        part = [(tuple(e["fit_ids"]), tuple(e["validation_ids"]),
                 [(tuple(z["fit_ids"]),tuple(z["holdout_ids"])) for z in e["calibration_CV"]]) for e in ev]
        if shared_partitions is None:
            shared_partitions = part
            for tr, va, cal in part:
                assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(manifest.index)
                assert all(set(u).isdisjoint(v) and set(u) | set(v) == set(tr) for u,v in cal)
            assert Counter(pid for _,va,_ in part for pid in va) == Counter(manifest.index.tolist())
        assert shared_partitions == part
        vals = []
        for e,m in zip(ev,folds):
            y = manifest.loc[e["validation_ids"], "LNM"]
            auroc = roc_auc_score(y,e["validation_probabilities"])
            assert np.isclose(auroc,m["AUROC"],rtol=0,atol=1e-14)
            assert len(e["gene_indices"]) == row.RNA_count
            genes[(row.selector,row.RNA_count,e["inner_fold"])] = e["gene_indices"]
            vals.append(m["AUROC"])
            rows.append({"selector":row.selector,"RNA_count":row.RNA_count,"candidate_id":row.candidate_id,
                         "fold":e["inner_fold"]+1,"validation_n":len(y),"validation_positive":int(y.sum()),
                         "AUROC":m["AUROC"],"trial_path":str(trial_path)})
        assert np.isclose(np.mean(vals),row.inner_mean_AUROC,rtol=0,atol=1e-14)
        assert np.isclose(np.std(vals,ddof=1),row.inner_SD_AUROC,rtol=0,atol=1e-14)
    assert all(genes[s,k,i] == genes[s,100,i][:k] for s in selectors for k in COUNTS for i in range(3))
    folds = pd.DataFrame(rows)
    keep = ["selector","RNA_count","candidate_id","parameters","attempt",
            "inner_mean_AUROC","inner_SD_AUROC","inner_mean_AUPRC"]
    heat = heat[keep].assign(n_inner_folds=3, final_recipe=lambda x: x.candidate_id.eq(selected_id))
    curve = heat[heat.selector.eq(SELECTOR)].copy().sort_values("RNA_count")
    for j in range(1,4):
        f = folds[(folds.selector == SELECTOR) & (folds.fold == j)].set_index("RNA_count")
        curve[f"fold_{j}_AUROC"] = curve.RNA_count.map(f.AUROC)
    ref = pd.read_csv(published_curve,float_precision="round_trip")
    assert np.array_equal(ref["count"],curve.RNA_count)
    assert np.array_equal(ref[keep[-3:]],curve[keep[-3:]])
    lr["used_in_fixed_heatmap"] = lr.stage.eq("A_complete_factorial") & lr.candidate_id.isin(heat.candidate_id)
    lr["is_frozen_candidate"] = lr.candidate_id.eq(selected_id)
    lr["is_first_appearance"] = ~lr.candidate_id.duplicated()
    save_csv(lr, "Stage21_LR_full_development_192_attempts")
    save_csv(folds, "Stage21_LR_fixed_parameters_162_inner_fold_records")
    save_csv(curve, NAMES[0])
    save_csv(heat, NAMES[1])
    # Secondary audit: cell maxima over the same three Stage A profiles, never Stage B.
    prof = a.sort_values(["inner_mean_AUROC","inner_mean_AUPRC","Brier","inner_SD_AUROC","candidate_id"],
                         ascending=[False,False,True,True,True]).drop_duplicates(["selector","RNA_count"])
    save_csv(prof[keep], "Stage21_StageA_best_of_equal_3_profiles_audit")
    hashes = {str(p): sha(p) for p in source}
    (OUT/"source_SHA256.json").write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding="utf-8")
    return curve,heat,selectors,lr,lock,hashes


def top(frame,title,note):
    frame.text(.5,frame.h-1.8,title,fontsize=8.5,fontweight="bold",va="top")
    frame.text(.5,frame.h-6.1,note,fontsize=7,va="top",color=MUTED)


def export(fig,name):
    for folder,ext in (("所有PDF小图","pdf"),("SVG备份","svg"),("PNG预览","png")):
        fig.savefig(PACKAGE/folder/f"{name}.{ext}",dpi=600)
    plt.close(fig)


def draw(curve,heat,selectors):
    fig = plt.figure(figsize=(110*MM,81*MM))
    fr = Frame(fig,(3,2,104,77))
    top(fr,"RNA feature count with the final selector and LR settings",
        "PLS-DA + logistic regression (L2, C = 1)  |  Mean ± SD across three inner folds")
    ax = fr.axes(left=12.5,bottom=12,right=4,top=13)
    x,y,s = (curve[c].to_numpy() for c in ("RNA_count","inner_mean_AUROC","inner_SD_AUROC"))
    ax.axhline(.5,color=MUTED,lw=.7,ls="--")
    ax.text(.02,.025,"Chance",transform=ax.transAxes,color=MUTED,fontsize=6.6)
    ax.errorbar(x,y,yerr=s,marker="o",ms=3.8,color=BLUE,lw=1.1,capsize=2.2,capthick=.85)
    ax.errorbar(x[-1],y[-1],yerr=s[-1],marker="o",ms=4.8,color=ORANGE,lw=1.15,capsize=2.2,capthick=.85,zorder=4)
    ax.annotate(f"Selected: {x[-1]}  |  AUROC {y[-1]:.3f}",(x[-1],y[-1]+s[-1]),
                xytext=(0,9),textcoords="offset points",ha="right",color=ORANGE,fontsize=7)
    ax.set(xlim=(0,106),ylim=(.5,1),xticks=COUNTS,yticks=np.arange(.5,1.01,.1),
           xlabel="Number of selected image-inferred RNA features",ylabel="Mean inner cross-validation AUROC")
    ax.grid(axis="y")
    export(fig,NAMES[0])

    fig = plt.figure(figsize=(122*MM,96*MM))
    fr = Frame(fig,(3,2,116,92))
    top(fr,"RNA selector and feature count with matched LR settings",
        "Balanced Stage A  |  Logistic regression (L2, C = 1)  |  Mean of three inner folds")
    ax = fr.axes(left=33,bottom=12,right=15,top=13)
    grid = heat.pivot(index="selector",columns="RNA_count",values="inner_mean_AUROC").loc[selectors,COUNTS].to_numpy()
    cmap = LinearSegmentedColormap.from_list("figure2_blues",["#F4F9FB","#B5D4E4",BLUE,"#16456C"])
    scale = (.5,.85)
    assert grid.min()>=scale[0] and grid.max()<=scale[1]
    mesh=ax.pcolormesh(grid,cmap=cmap,vmin=scale[0],vmax=scale[1],edgecolors="white",linewidth=.5)
    ax.set(xticks=np.arange(len(COUNTS))+.5,xticklabels=COUNTS,
           yticks=np.arange(len(selectors))+.5,yticklabels=selectors,
           xlabel="Number of selected image-inferred RNA features")
    ax.invert_yaxis()
    for i in range(len(selectors)):
        for j in range(len(COUNTS)):
            v=grid[i,j]
            ax.text(j+.5,i+.5,f"{v:.3f}",ha="center",va="center",fontsize=7,
                    color="white" if v>=.71 else INK)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    ax.add_patch(Rectangle((5.02,selectors.index(SELECTOR)+.02),.96,.96,fill=False,
                           linewidth=1.35,edgecolor=ORANGE,zorder=5))
    pos=ax.get_position()
    cax=fig.add_axes([pos.x1+.022,pos.y0,.023,pos.height])
    bar=fig.colorbar(mesh,cax=cax,ticks=[.50,.60,.70,.80,.85])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.ax.set_ylabel("Mean inner AUROC",labelpad=3)
    export(fig,NAMES[1])


def document(curve,heat,lr,lock,hashes):
    c = curve.set_index("RNA_count")
    delta = c.loc[100,"inner_mean_AUROC"] - c.loc[50,"inner_mean_AUROC"]
    diff = [c.loc[100,f"fold_{j}_AUROC"]-c.loc[50,f"fold_{j}_AUROC"] for j in range(1,4)]
    assert all(z>0 for z in diff)
    b = lr[lr.stage.eq("B_all_classifiers_refined")].sort_values("inner_mean_AUROC",ascending=False).iloc[0]
    legends = f"""**RNA feature count with the final selector and logistic regression settings.** Points show mean AUROC and error bars indicate sample standard deviation across the same three patient-level inner validation folds of the full surgical development cohort (160 patients; fold sizes 54, 53 and 53). PLS-DA rankings were estimated within the corresponding training partitions. The six candidate feature counts were compared using the same L2-penalized logistic regression settings (C = 1) and training-only sigmoid calibration procedure. The orange point identifies the frozen Top100 recipe (AUROC {c.loc[100,'inner_mean_AUROC']:.4f}); Top50 yielded {c.loc[50,'inner_mean_AUROC']:.4f}. The difference of {delta:.4f} is descriptive of the training search and does not establish statistical superiority. The dashed line indicates chance discrimination.

**RNA selector and feature count with matched logistic regression settings.** The 54 cells come from the balanced first stage of the full-development training search. Each cell represents one selector–count combination evaluated using the same logistic regression parameters (L2 penalty, C = 1), the same three inner patient splits and the same calibration procedure. Cell labels are the unweighted mean of the three fold AUROCs. No per-cell hyperparameter maximization or adaptively refined trials were used. The orange outline marks the frozen PLS-DA Top100 recipe. These are training-internal selection results; the outer held-out, ESD and TCGA data were not used for selection.

AUROC, area under the receiver operating characteristic curve; LR, logistic regression; PLS-DA, partial least squares discriminant analysis; SD, standard deviation. RNA features were inferred from images.
"""
    (PACKAGE/"图注/RNA_feature_optimization_legends.md").write_text(legends,encoding="utf-8")
    table = "\n".join(f"| {k} | {r.inner_mean_AUROC:.6f} | {r.inner_SD_AUROC:.6f} | " +
                     " | ".join(f"{r[f'fold_{j}_AUROC']:.6f}" for j in range(1,4))+" |"
                     for k,r in c.iterrows())
    report = f"""# 第21阶段：为何最终选择100个图像推断RNA特征

在外科全开发集的三折训练内搜索中，PLS-DA Top100 + L2逻辑回归（C=1）获得最高平均AUROC **{c.loc[100,'inner_mean_AUROC']:.6f}**。它在均衡第一阶段第126次尝试已是第一，细化阶段最佳组合为{b.selector} Top{b.RNA_count}（{b.inner_mean_AUROC:.6f}），未超过它。第三阶段在同一筛选器、同一参数下逐一核对六个数量；第192次是同一配方记录，并非独立重复验证。192次尝试全部有效，186个唯一配方。核对完整汇总日志（13,824次，含全开发集12个分类器的2,304次）及冻结排名：最终配方的全开发集内层平均AUROC为最高，未动用次级平局规则。外层、ESD、TCGA分数均未进入这项选择。参见 `train_all_12.py:98`、`:109`、`:125`、`:153`、`:161`、`:217`。

| RNA特征数量 | 平均内层AUROC | 折间样本SD | 折1（n=54） | 折2（n=53） | 折3（n=53） |
| ---: | ---: | ---: | ---: | ---: | ---: |
{table}

100个比50个平均高 **{delta:.9f} AUROC**（{delta*100:.3f}个百分点）；三个配对折差值依次为 {', '.join(f'+{z:.5f}' for z in diff)}。在这六个匹配候选中，Top100的均值和三个单折AUROC均最高。折间SD是三折分散程度，不是标准误或置信区间；三折训练样本重叠，且100是根据同一轮搜索选出的，这些记录不能证明显著或可泛化的优势。100是测试上限，但胜出依据是实际分数；日志没有100以上的候选，无法判断更大数量是否更好。

数量曲线固定PLS-DA筛选器、LR L2惩罚、C=1、训练划分和校准方法。每折各数量的基因列表均是同一排名的前缀；模型在各自训练部分拟合。曲线可描述固定配方族中数量变化的表现。正式冻结是筛选器、数量和超参数的联合搜索，不能把跨配方差异归因为数量。该图足以说明为何在本次候选范围和规则下冻结100，不足以证明100是普遍最佳数量。

热图固定C=1和L2惩罚，使用均衡第一阶段9×6共54个单元格；不逐格挑参，也不混入只覆盖两种筛选器的细化阶段。所有单元格核对了完全相同的三折患者分配和校准子折，从保存的验证概率复算了折AUROC。三种共同参数档的完整记录另附；各格在均衡三档中取最高值的统计只在审计CSV中提供。

冻结模型 SHA256：`{lock['model_SHA256']}`。本次只读保存的训练结果和验证概率，不重新训练或更新模型。图稿使用第23阶段Figure 2的Arial和矢量设置。
"""
    (OUT/"RNA_feature_optimization_audit.md").write_text(report,encoding="utf-8")
    pd.DataFrame([{"file":n+".pdf","cohort":"Surgical full-development inner CV","model":"Stage21 clinical_LNM_v2, logistic regression",
                   "source_csv":str(OUT/(n+".csv")),"pdf":str(PACKAGE/"所有PDF小图"/(n+".pdf")),
                   "svg":str(PACKAGE/"SVG备份"/(n+".svg")),"png":str(PACKAGE/"PNG预览"/(n+".png"))}
                  for n in NAMES]).to_csv(OUT/"supplemental_panel_index.csv",index=False,encoding="utf-8-sig")
    assert all(sha(Path(p))==h for p,h in hashes.items())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = extract()
    draw(*data[:3])
    document(data[0],data[1],*data[3:])
