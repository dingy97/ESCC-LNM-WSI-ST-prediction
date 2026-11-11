"""Verify vector exports and build deidentified review sheets and captions."""
from collections import Counter
from contextlib import closing
import json
import sys
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import openslide

from stage24_attribution import OUT, DATA, QA, PRIVATE, FOLDERS, read, sha, table
from stage24_draw import PANELS, footprint, read_csv

FONT=r"C:\Windows\Fonts\arial.ttf"


def contact(items, dest, *, cols=4, cell=(400,345)):
    w,h=cell
    canvas=Image.new("RGB",(cols*w,((len(items)+cols-1)//cols)*h), "#EDF1F4")
    dr=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,14)
    for i,(im,label) in enumerate(items):
        x,y=i%cols*w,i//cols*h
        dr.rectangle((x+4,y+4,x+w-5,y+h-5),fill="white")
        im.thumbnail((w-16,h-28),Image.Resampling.LANCZOS)
        canvas.paste(im,(x+(w-im.width)//2,y+7+(h-28-im.height)//2))
        dr.text((x+10,y+h-21),label,fill="#182632",font=font)
    canvas.save(dest)


def exports():
    fov=read_csv(DATA/"all_WSI_FOV_attribution.csv")
    gate=read(QA/"attribution_gate.json")
    scale=read(QA/"plotting_contract.json")["raw_margin_delta_absolute_limit"]
    assert gate["status"]=="PASS" and len(fov)==gate["FOVs"]
    valid=fov.attribution_status.eq("PASS")
    z=fov.loc[valid]
    assert np.array_equal(z.full_raw_margin-z.without_FOV_raw_margin,z.delta_raw_margin)
    assert np.allclose(z.full_calibrated_probability-z.without_FOV_calibrated_probability,
                       z.delta_calibrated_probability,atol=1e-16,rtol=0)
    assert np.array_equal(np.sign(z.delta_raw_margin),np.sign(z.delta_calibrated_probability))
    assert z.delta_raw_margin.abs().max()<=scale
    assert fov.retained_patches.between(1,16).all()
    qa=[]
    for group,folder in FOLDERS.items():
        images=[]
        for name in PANELS:
            stem=OUT/folder/name
            pdf=stem.with_suffix(".pdf")
            with fitz.open(pdf) as doc:
                assert doc.page_count==1
                page=doc[0]
                paths=len(page.get_drawings())
                fonts=page.get_fonts()
                text=page.get_text()
                pic=page.get_image_info()
                assert paths>0 and len(pic)==1 and text and fonts
                assert all(font[2]!="Type3" and doc.extract_font(font[0])[-1] for font in fonts)
                for b in page.get_text("dict",flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
                    for l in b.get("lines",[]):
                        for s in l["spans"]: assert page.rect.contains(fitz.Rect(s["bbox"]))
                if name=="LNM_attribution_heatmap":
                    assert "leave-one-FOV-out" in text and f"+{scale:.3f}" in text
                pix=page.get_pixmap(matrix=fitz.Matrix(2.6,2.6),alpha=False)
                images.append((Image.frombytes("RGB",(pix.width,pix.height),pix.samples),name))
            tags=Counter(x.tag.split("}")[-1] for x in ET.parse(stem.with_suffix(".svg")).iter())
            assert tags["image"]==1 and tags["text"]>0 and tags["path"]>0
            with Image.open(stem.with_suffix(".png")) as im:
                dpi=im.info["dpi"]
                assert all(abs(z-600)<.1 for z in dpi)
                im.verify()
            qa.append({"group":group,"panel":name,"PDF_pages":1,"PDF_vector_objects":paths,"PDF_HE_images":len(pic),
                       "HE_pixels_x":pic[0]["width"],"HE_pixels_y":pic[0]["height"],"embedded_font_resources":len(fonts),
                       "SVG_editable_text":tags["text"],"PNG_dpi":round(dpi[0]),"SHA256":sha(pdf)})
        contact(images,QA/f"{group}_panels_review.jpg",cols=2,cell=(630,590))
    table(pd.DataFrame(qa),QA/"PDF_SVG_PNG_verification.csv")


def all_slide_review():
    slides=read_csv(DATA/"anonymous_slide_QC.csv")
    fovs=read_csv(DATA/"all_WSI_FOV_attribution.csv")
    private=read_csv(PRIVATE/"slide_identity_and_source.csv").set_index("anonymous_slide_id")
    for g in FOLDERS:
        images=[]
        for s in slides.loc[slides.group.eq(g)].itertuples():
            d=fovs[fovs.anonymous_slide_id.eq(s.anonymous_slide_id)]
            assert len(d)==s.valid_FOVs
            p=private.loc[s.anonymous_slide_id,"source_path"]
            with closing(openslide.OpenSlide(p)) as w:
                thumb=w.get_thumbnail((750,750)).convert("RGB")
                sx,sy=np.asarray(thumb.size)/np.asarray(w.dimensions)
                roi=d.loc[d.delta_raw_margin.abs().idxmax()] if s.attribution_FOVs else d.iloc[0]
                x,y,ww,hh=footprint(roi)
                detail=w.read_region((int(x),int(y)),0,(int(ww),int(hh))).convert("RGB")
            draw=ImageDraw.Draw(thumb)
            for r in d.itertuples():
                x,y,ww,hh=footprint(r)
                draw.rectangle((x*sx,y*sy,(x+ww)*sx,(y+hh)*sy),
                    outline="#146A85",width=2)
            x,y,ww,hh=footprint(roi)
            draw.rectangle((x*sx,y*sy,(x+ww)*sx,(y+hh)*sy),outline="#FFB31C",width=3)
            both=Image.new("RGB",(670,325),"white")
            thumb.thumbnail((365,314),Image.Resampling.LANCZOS)
            detail.thumbnail((284,280),Image.Resampling.LANCZOS)
            both.paste(thumb,((365-thumb.width)//2,(325-thumb.height)//2))
            both.paste(detail,(377+(284-detail.width)//2,(325-detail.height)//2))
            s0="Main · " if s.main_slide else ""
            images.append((both,f"{s0}{s.anonymous_slide_id}  |  {s.valid_FOVs} FOVs"))
        contact(images,QA/f"{g}_all_WSI_footprints_and_detail.jpg",cols=2,cell=(700,345))


def illustrator():
    import win32com.client
    app=win32com.client.Dispatch("Illustrator.Application")
    result=[]
    for group,folder in FOLDERS.items():
        for name in PANELS:
            p=OUT/folder/(name+".pdf")
            before=sha(p)
            js="""
var previous = app.userInteractionLevel;
var doc = null;
var result = '';
try {
  app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
  doc = app.open(new File(FILE));
  result = [doc.pathItems.length,doc.textFrames.length,doc.rasterItems.length,doc.placedItems.length].join('|');
} finally {
  if (doc) doc.close(SaveOptions.DONOTSAVECHANGES);
  app.userInteractionLevel = previous;
}
result;
""".replace("FILE",json.dumps(p.as_posix(),ensure_ascii=True))
            paths,text,rasters,placed=map(int,app.DoJavaScript(js).split("|"))
            assert paths>0 and text>0 and sha(p)==before
            result.append({"group":group,"panel":name,"editable_paths":paths,"editable_text_frames":text,
                           "raster_items":rasters,"placed_items":placed,"SHA256":before})
            print(group,name,paths,text,rasters,placed,flush=True)
            table(pd.DataFrame(result),QA/"Illustrator_import_QA.csv")
    assert len(result)==16


def report():
    sel=read_csv(DATA/"anonymous_case_selection.csv")
    f=read_csv(DATA/"all_WSI_FOV_attribution.csv")
    s=read_csv(DATA/"anonymous_slide_QC.csv")
    main=read_csv(QA/"figure_geometry_and_coverage.csv").set_index("group")
    roi=read_csv(DATA/"displayed_ROI.csv").set_index("group")
    rep=read_csv(QA/"all_59_FOV_to_patient_exact_replay.csv")
    m=read_csv(QA/"selected_case_margin_and_probability_replay.csv")
    plot=read(QA/"plotting_contract.json")
    gate=read(QA/"attribution_gate.json")
    visual=read(QA/"visual_review_notes.json")
    ai=read_csv(QA/"Illustrator_import_QA.csv")
    assert len(ai)==16 and ai.editable_paths.gt(0).all() and ai.editable_text_frames.gt(0).all()
    assert len(rep)==59 and rep.WSI_count.sum()==749 and rep.all_6175_features_bit_exact.all()
    line=[]
    distribution=[]
    legend_list=[]
    for case in sel.itertuples():
        g=case.group
        d=f[f.group.eq(g)]
        w=s[s.group.eq(g)]
        r=roi.loc[g]
        p=main.loc[g]
        counts=((d.delta_raw_margin>0).sum(),(d.delta_raw_margin<0).sum(),(d.delta_raw_margin==0).sum())
        with_both=(w.positive_FOVs.gt(0)&w.negative_FOVs.gt(0)).sum()
        line.append(f"| {g} | {case.anonymous_case_id} | {case.corrected_LNM} | {case.archived_probability:.9f} | "
            f"{case.WSI_count} / {case.total_valid_FOVs} | {case.main_slide} / {p.valid_FOVs} | "
            f"{p.tissue_area_mm2:.2f} | {p.approx_sampled_tissue_fraction:.1%} |")
        distribution.append(f"- **{g}**。全病例正向／负向／零贡献为 {counts[0]}／{counts[1]}／{counts[2]} 个 FOV，"
            f"Δ原始对数优势范围 {d.delta_raw_margin.min():+.5f} 至 {d.delta_raw_margin.max():+.5f}，"
            f"{with_both}/{len(w)} 张切片同时出现正负贡献。主切片放大 FOV {int(r.FOV_ordinal)} 的 Δ为 {r.delta_raw_margin:+.5f}，"
            f"校准概率变化 {r.delta_calibrated_probability:+.6f}。掩膜支持 {r.candidate_mask_fraction:.1%}，"
            f"原流程使用 {int(r.retained_patches)}/16 个图块。{visual[g]}")
        legend_list.append(f"- **{g} ({case.anonymous_case_id}).** Corrected LNM status {case.corrected_LNM}, frozen probability "
            f"{case.archived_probability:.3f}, {case.WSI_count} slides and {case.total_valid_FOVs} valid virtual FOVs. "
            f"The illustrated slide contains {p.valid_FOVs} valid FOVs. The native detail shows FOV {int(r.FOV_ordinal)} "
            f"(Δ uncalibrated log-odds {r.delta_raw_margin:+.4f}). Candidate-mask support occupies "
            f"{r.candidate_mask_fraction:.1%} of the displayed FOV and {int(r.retained_patches)} of 16 tiles entered the model.")
    notes=f"""# 第24阶段｜真实 H&E 病理风险归因

已完成四个代表病例的 16 张独立图。主文为 TP 与唯一 FN，补充材料为 TN 与 FP。纠正后主队列的 TP／FN／TN／FP 分别为 53／1／2／3，阈值为 {gate['score_threshold']:.17g}。

| 类别 | 匿名病例 | 纠正LNM | 冻结概率 | 全病例 WSI／有效FOV | 主切片／有效FOV | 主切片组织支持面积 mm² | 主切片估计采样覆盖 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
{chr(10).join(line)}

病例在查看归因前锁定。TP、TN、FP 按各组预测概率距精确十进制中位数的最小距离选择，TN 并列按匿名 ID 字典序。FN 是唯一的漏诊病例。主切片取冻结流程组织支持掩膜面积最大者，面积相同按匿名切片 ID 排序。没有独立肿瘤面积标注，掩膜面积表示组织支持。放大区域为主切片内绝对贡献最大的已采样 FOV，数值相同按 FOV 序号排序。各组技术回退记录见 QA。

冻结流程在水平和垂直物理尺度约 512 µm 的虚拟 FOV 内将合格 H&E 图块的 Phikon CLS 向量取均值，应用冻结 CosMx 监督 Ridge，产生 6,175 个图像推断 RNA 特征。先在每张 WSI 内按有效 FOV 等权取均值，再在病理号组内按 WSI 等权取均值。患者向量按已保存的 PLS-DA 排名索引取100列，经过已冻结的标准化器，进入 LR，再将 LR 的原始 log-odds 送入训练期交叉拟合的固定 sigmoid 校准器。参见第21阶段原提取脚本、纠正分支的 `prepare_esd_validation.py` 与历史 `rna_experiment.CalibratedRNAModel`。

59个病理号组的 6,175 维聚合特征均从 749 张归档 WSI 缓存逐位复现。归档概率的最大绝对误差为 {rep.absolute_error.max():.3g}，分类完全一致。冻结模型 SHA256 为 `{gate['model_sha256']}`。校准斜率为 {m.calibration_slope.iloc[0]:.6f}，因此校准后概率变化与原始 log-odds 变化方向一致。两种变化均在归因 CSV 中保存。

每次移除一个有效 FOV，先重算其切片的 FOV 均值，再按原切片权重重算患者特征，并用冻结流程计算分数。Δ为完整减移除后的原始 LR log-odds。热图统一对称色标 ±{plot['raw_margin_delta_absolute_limit']:.3f}，红色表示存在该 FOV 时分数较高，蓝色表示较低。叠加不透明度为归一化绝对贡献的平方根乘 0.76，四例相同。最终模型采用等权聚合，缺少 LNM 专用 attention 权重。归因反映该 FOV 相对同切片其余采样区域的影响，受同病例切片数与该片 FOV 数影响，各 FOV 的 Δ 不构成独立的可加患者风险分解。

归因用原流程的 float32 均值重算，通过独立 LR 权重点积核对。理想 float64 公式与实际重算之间最大舍入差为 {f.float32_aggregation_roundoff_delta.abs().max():.6g} log-odds，本次没有改变任何 FOV 的贡献符号。所有单 FOV 移除后，病例仍在原冻结阈值的同一侧。

{chr(10).join(distribution)}

目视核查覆盖32张切片的足迹总览及每张的极值FOV原片局部。未对全部1,024个FOV作病理医师逐框复阅。主文可将 TP 和 FN 的原图、足迹与归因并列。若展示原倍率局部图，应保留其组织覆盖及边缘、杂质说明。TN 与 FP 按预定用途放补充材料。

所有绘图坐标为原片 Level 0 像素。旧缓存由重新执行原采样及图块组织筛选验证坐标和逐图块组织比例。FP 新缓存缺少坐标，按同一采样、Phikon FP32 和 Ridge 重新推理后，与归档逐 FOV RNA 行逐位比对，通过后使用恢复坐标。{int((~s.quickhash_available).sum())} 张 Aperio 切片在归档及本次读取时均无 OpenSlide Quickhash，身份核验使用原路径、文件字节数、修改时间、尺寸、MPP 和逐 FOV 坐标与图块组织比例的完全一致性。四病例的全部切片归因已保留。组织支持内上色，其他位置保留原始 H&E。估计采样组织覆盖来自冻结缩略图掩膜，表示 FOV 足迹覆盖的支持比例。FOV 内只有达到条件的图块实际用于特征计算，稀疏或未采样区域的风险无法从这些图判定。

16 份独立 PDF 已在本机 Adobe Illustrator 实际导入，均保留可编辑文字和路径。H&E 图像作为原片位图嵌入。另有可编辑 SVG 和 600 dpi PNG。病例身份和原切片路径仅在本地 `QA/Private` 留存。依次运行 `Scripts/stage24_attribution.py`、`Scripts/stage24_draw.py` 和 `Scripts/stage24_verify.py exports review illustrator report` 可复核流程。以上属探索性的模型解释，不能用来认定组织学转移机制。
"""
    (OUT/"Reports/stage24_LNM_attribution_heatmap_report.md").write_text(notes,encoding="utf-8")
    legends=f"""**Exploratory H&E attribution for lymph node metastasis prediction in representative ESD cases.**
Cases were selected within the corrected-label ESD cohort by proximity to the group median frozen probability, with exact ties resolved by anonymous identifier. The sole false negative was included. The primary slide had the largest tissue-support area under the frozen sampling mask. TP and FN are main-text examples. TN and FP are supplementary examples.

{chr(10).join(legend_list)}

**A,** Original H&E at the overview scale. **B,** Full footprints of the virtual FOVs that contributed features. **C,** Leave-one-FOV-out attribution on the same H&E overview. The frozen slide mean and equally weighted case mean were recomputed for each deletion. The shared diverging scale spans ±{plot['raw_margin_delta_absolute_limit']:.3f} in uncalibrated LR log-odds. Red indicates that the field increased the score relative to omitting it, and blue indicates a decrease. Color is shown only within sampled footprints intersecting the candidate tissue mask. Opacity scales with the square root of the normalized absolute change and is shown on a white reference in the color bar. **D,** Native Level 0 H&E of the FOV with the largest absolute attribution on the illustrated slide, marked by the dashed amber box in B and C. The four selected FOVs include tissue edges and background. Scale bars are 1 mm for the overview and 100 µm for the detail. The frozen case-level cutoff was {gate['score_threshold']:.9f}. These sparse local perturbations describe the model's prediction and are exploratory.

ESD, endoscopic submucosal dissection. FOV, field of view. FN, false negative. FP, false positive. LR, logistic regression. TN, true negative. TP, true positive.
"""
    (OUT/"Legends/stage24_English_figure_legends.md").write_text(legends,encoding="utf-8")


if __name__=="__main__":
    tasks={"exports":exports,"review":all_slide_review,"illustrator":illustrator,"report":report}
    for arg in sys.argv[1:]: tasks[arg]()
