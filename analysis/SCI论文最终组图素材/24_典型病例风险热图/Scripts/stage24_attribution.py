"""Audit the frozen ESD prediction and map leave-one-FOV-out changes."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from contextlib import closing
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(r"__ESCC_PROJECT_ROOT__")
STAGE = ROOT / "21_surgical_RNA_LNM_development"
V21 = STAGE / "clinical_LNM_v2"
ESD = V21 / "04_ESD_validation"
OLD = STAGE / "04_ESD_validation"
PKG = ROOT / "SCI论文最终组图素材"
OUT = PKG / "24_典型病例风险热图"
DATA, QA, PRIVATE = OUT / "Source_Data", OUT / "QA", OUT / "QA/Private"
FOLDERS = {"TP": "Main_TP", "FN": "Main_FN", "TN": "Supplementary_TN", "FP": "Supplementary_FP"}
SAMPLER = ROOT / "10_lnm_sampling_corrected/scripts"
BRIDGE = ROOT / "03_bridge_training/bridge_model_checkpoints/broad_transcriptome"
SAMPLING = ROOT / "10_lnm_sampling_corrected/config/frozen_analysis.json"
OLD_WSIS = STAGE / "00_inventory/ESD_WSI_manifest.csv"
PRED = ESD / "ESD_unseen_pipeline_case_groups_frozen_predictions.csv"
LABEL = PKG / "原始数据索引/ESD_case_label_correction_INTERNAL.csv"
READY = ESD / "ESD_features_ready.json"
LOCK = V21 / "config/final_model_lock.json"
PARQUET = ESD / "ESD_case_inferred_RNA.parquet"
sys.path.insert(0, str(ROOT / "08_lnm_modeling/runtime_deps"))
import numpy as np
import pandas as pd
import joblib
import openslide
from threadpoolctl import threadpool_limits
from PIL import Image
from scipy.special import expit


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def table(frame, file):
    frame.to_csv(file, index=False, encoding="utf-8-sig", float_format="%.17g")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def code(s):
    return hashlib.sha256(s.encode()).hexdigest()


def check_quickhash(slide, archived):
    current = slide.properties.get("openslide.quickhash-1")
    assert current == (None if pd.isna(archived) else archived)
    return current is not None


def initialize():
    for d in [DATA, QA, PRIVATE, OUT/"Reports", OUT/"Legends", OUT/"Scripts", *(OUT/x for x in FOLDERS.values())]:
        d.mkdir(parents=True, exist_ok=True)
    # Includes unredacted internal mapping and raw source paths. Keep local.
    private = PRIVATE / "recovered_geometry"
    private.mkdir(exist_ok=True)


def audit_and_select():
    lock = read(LOCK)
    model_path = Path(lock["winner_directory"]) / "model.joblib"
    assert sha(model_path) == lock["model_SHA256"]
    assert lock["winner_classifier"] == "Logistic Regression"
    assert lock["winner"]["recipe"] == {"selector":"PLS-DA","count":100,"config":{"C":1.0,"penalty":"l2"}}
    extract_contract = read(OLD/"extraction_complete.json")["contract"]
    ready = read(READY)
    assert extract_contract == ready["feature_contract"]
    check = {"script":STAGE/"scripts/extract_ESD_features.py", "frozen_sampling":SAMPLING,
             "sampler":SAMPLER/"spatial_sampling.py", "bridge":BRIDGE/"Phikon_FULL_B_Ridge.joblib"}
    assert all(sha(path)==extract_contract[key] for key,path in check.items())
    assert sha(V21/"scripts/prepare_esd_validation.py")==ready["added_extractor_sha256"]
    assert sha(ESD/"patient_LNM_folder_authority.json")==ready["authority_sha256"]
    assert all(sha(ESD/name)==digest for name,digest in ready["outputs"].items())
    audit = read(PKG/"原始数据索引/source_audit.json")["source_SHA256"]
    assert audit[PRED.relative_to(ROOT).as_posix()]==sha(PRED)

    pred = pd.read_csv(PRED,dtype={"probability":str,"case_id":str})
    labels = pd.read_csv(LABEL).set_index("Patient_ID")
    primary = pd.read_csv(ESD/"ESD_primary_analysis_manifest.csv",dtype={"case_id":str}).set_index("Patient_ID")
    assert len(pred)==59 and pred.Patient_ID.is_unique and set(pred.Patient_ID)==set(primary.index)
    assert np.array_equal(primary.loc[pred.Patient_ID,"case_id"],pred.case_id)
    assert np.array_equal(primary.loc[pred.Patient_ID,"LNM"],pred.LNM)
    assert np.array_equal(labels.loc[pred.Patient_ID,"original_LNM"],pred.LNM)
    pred["corrected"] = labels.loc[pred.Patient_ID,"corrected_LNM"].to_numpy()
    assert np.array_equal(pred.corrected,1-pred.LNM) and pred.corrected.sum()==54
    assert not pred[["development_overlap","bridge_overlap"]].any().any()
    assert pred.model_SHA256.eq(lock["model_SHA256"]).all()
    pred["decimal"] = pred.probability.map(Decimal)
    cut = lock["winner"]["thresholds"]["0.98"]
    pred["high"] = pred.decimal.ge(Decimal(str(cut)))
    assert np.array_equal(pred.high,pred["high_risk_0.98"])
    pred["group"] = np.where(pred.corrected==1,np.where(pred.high,"TP","FN"),np.where(pred.high,"FP","TN"))
    assert pred.group.value_counts().to_dict()=={"TP":53,"FN":1,"FP":3,"TN":2}
    pred["anonymous_case_id"] = pred.Patient_ID.map(lambda v:"ESD-"+code("stage24-20261108:"+v)[:10].upper())
    assert pred.anonymous_case_id.is_unique

    sys.path.insert(0,str(ROOT/"14_RNA_LNM_classifier_optimization/scripts"))
    import rna_experiment
    model = joblib.load(model_path)
    assert model.estimator.__class__.__name__=="LogisticRegression"
    assert model.calibrator.__class__.__name__=="LogisticRegression"
    assert len(model.features.genes)==100 and len(set(model.features.genes))==100
    genes = read(BRIDGE/"bundle_card.json")["gene_names"]
    assert code("|".join(genes))==extract_contract["gene_order"]
    matrix = pd.read_parquet(PARQUET)
    assert list(matrix.columns)==genes and matrix.index.is_unique
    assert [genes[i] for i in model.features.genes]==lock["winner"]["selected_genes"]
    arr = matrix.loc[pred.Patient_ID].to_numpy(float)
    scores = model.predict_proba(arr)[:,1]
    error = np.abs(scores-pred.probability.astype(float).to_numpy())
    assert error.max()<=1e-12, f"Frozen scores not reproduced: max difference {error.max()}"
    assert np.array_equal(scores>=cut,pred.high)
    table(pred[["anonymous_case_id","corrected","group","probability"]].assign(
        reproduced_probability=scores,absolute_error=error),QA/"cohort_prediction_reproduction.csv")

    # Fixed group order and exact-decimal median distance. Names cannot influence order.
    ordering = []
    for group in FOLDERS:
        p = pred.loc[pred.group.eq(group)].copy()
        a = sorted(p.decimal)
        median = (a[(len(a)-1)//2]+a[len(a)//2])/2
        p["distance"] = p.decimal.map(lambda x:abs(x-median))
        p["median"] = str(median)
        p = p.sort_values(["distance","anonymous_case_id"])
        p["selection_rank"] = np.arange(1,len(p)+1)
        ordering.append(p)
    ordered = pd.concat(ordering,ignore_index=True)
    columns = ["group","anonymous_case_id","corrected","probability","median","distance","selection_rank","usable_WSI_count"]
    public = ordered[columns]
    candidate_lock = QA/"case_selection_lock.csv"
    if candidate_lock.exists():
        assert pd.read_csv(candidate_lock,dtype=str).equals(public.astype(str).reset_index(drop=True))
    else:
        table(public,candidate_lock)
    table(ordered[columns+["Patient_ID","case_id"]],PRIVATE/"case_identity_and_selection.csv")
    write(QA/"selection_protocol.json", {
        "group_sizes":{"TP":53,"FN":1,"TN":2,"FP":3}, "threshold":cut,
        "case_rule":"Within each group minimize exact decimal distance to group median; ties by ascending anonymous case ID.",
        "anonymous_id_rule":"ESD- plus first 10 upper-case SHA256 hex characters of a fixed Stage24 prefix and archived patient key.",
        "technical_fallback":"Only unreadable or no valid tissue can advance to the next fixed group rank. Cache identity/replay/coordinate failures halt.",
        "slide_rule":"Among all frozen usable WSI, greatest frozen candidate tissue mask area in mm2; ties by anonymous slide ID. No tumor outlines available.",
        "ROI_rule":"On the main slide choose FOV with largest absolute raw margin delta; ties by archived FOV ordinal. Show that FOV at native resolution.",
        "selection_occurred_before_attribution":True, "model_SHA256":lock["model_SHA256"],
        "archived_score_max_absolute_error":float(error.max()),
        "input_SHA256":{p.name:sha(p) for p in (PRED,LABEL,LOCK,PARQUET,READY,OLD_WSIS)},
    })
    print("Frozen model replayed for 59 case groups. Max probability error",error.max(),flush=True)
    return lock,model,matrix,ordered,extract_contract


def get_inputs(row,contract):
    p=Path(row.path)
    stat=p.stat()
    assert stat.st_size==row.bytes and str(stat.st_mtime_ns)==str(row.mtime_ns)
    assert p.is_relative_to(Path(r"__ESCC_ESD_WSI_ROOT__"))
    key=code(row.path)[:24]
    if row.source=="original_CUDA_cache":
        cache=OLD/"features"/(key+".npz")
        meta=read(cache.with_suffix(".json"))
        assert meta["status"]=="PASS" and meta["contract"]==contract
        assert meta["path"]==row.path and meta["Patient_ID"]==row.Patient_ID and meta["case_id"]==row.case_id
        assert meta["source_bytes"]==row.bytes and meta["source_mtime_ns"]==str(row.mtime_ns)
    else:
        assert row.source=="new_CUDA_features"
        cache=ESD/"added_WSI_features"/(key+".npz")
        meta=None
    with np.load(cache) as f:
        stored={k:f[k] for k in f.files}
    rna=stored["RNA"]
    assert rna.dtype==np.float32 and rna.ndim==2 and 1<=len(rna)<=32 and rna.shape[1]==6175
    assert np.isfinite(rna).all()
    if meta: assert meta["valid_FOVs"]==len(rna)
    return key,cache,meta,stored


def geometry(row, inputs, contract, rebuild):
    # Re-run the exact patch-admission test without encoding the cached old slides.
    from extract_wsi_features import slide_candidate_mask, candidate_fovs, read_patches
    key,cache,meta,stored=inputs
    saved = PRIVATE / "recovered_geometry" / (key+".npz")
    cache_hash=sha(cache)
    if saved.exists():
        with np.load(saved) as f: g={k:f[k] for k in f.files}
        assert str(g["cache_sha"])==cache_hash and str(g["sampler_sha"])==contract["sampler"]
        assert str(g["extractor_sha"])==sha(SAMPLER/"extract_wsi_features.py")
        assert str(g["source_mtime_ns"])==str(row.mtime_ns) and int(g["source_bytes"])==row.bytes
        if meta:
            np.testing.assert_array_equal(g["coords"],stored["coords_level0"])
            np.testing.assert_array_equal(g["fractions"],stored["patch_tissue_fractions"])
        else: np.testing.assert_array_equal(g["reproduced_RNA"],stored["RNA"])
        return g
    cfg=read(SAMPLING)
    with closing(openslide.OpenSlide(row.path)) as slide:
        mpp=np.array([float(slide.properties[k]) for k in (openslide.PROPERTY_NAME_MPP_X,openslide.PROPERTY_NAME_MPP_Y)])
        assert np.array_equal(slide.dimensions,[row.width,row.height])
        np.testing.assert_allclose(mpp,[row.MPP_X,row.MPP_Y],rtol=0,atol=1e-12)
        check_quickhash(slide,row.quickhash)
        thumb=np.asarray(slide.get_thumbnail((2048,2048)).convert("RGB"))
        mask,_=slide_candidate_mask(thumb,slide.dimensions,mpp)
        candidates,qc=candidate_fovs(slide.dimensions,mpp,mask,cfg["FOV_um"],cfg["max_virtual_FOVs_per_slide"])
        images,groups,fractions=read_patches(slide,candidates,mpp,cfg["FOV_um"])
    if not len(images):
        raise ValueError("No tissue-positive patches under the frozen sampler")
    valid=np.unique(groups)
    coords=candidates[valid]
    fractions=fractions[valid]
    fw,fh=np.rint(np.asarray(cfg["FOV_um"])/mpp).astype(int)
    assert len(coords)==len(stored["RNA"])
    assert (coords>=0).all() and (coords+[fw,fh]<=[row.width,row.height]).all()
    assert np.array_equal((fractions>=.05).sum(1),np.bincount(groups,minlength=len(candidates))[valid])
    packed={}
    if meta:
        np.testing.assert_array_equal(coords,stored["coords_level0"])
        np.testing.assert_array_equal(fractions,stored["patch_tissue_fractions"])
        assert len(images)==meta["patches"] and qc==meta["sampling"]
        method="Archived coordinates and all patch fractions reproduced exactly"
    else:
        encoded=rebuild(images,groups,valid)
        np.testing.assert_array_equal(encoded,stored["RNA"],err_msg="Reconstructed geometry must reproduce archived ordered RNA rows")
        packed["reproduced_RNA"]=encoded
        method="Restored deterministic coordinates with exact archived RNA row equality"
    scale=np.array([row.width,row.height])*mpp/np.array(mask.shape[::-1])
    tissue_mm2=mask.sum()*np.prod(scale)/1e6
    cover=[]
    for x,y in coords:
        a=np.floor(np.array([x,y])*np.array(mask.shape[::-1])/[row.width,row.height]).astype(int)
        b=np.ceil((np.array([x,y])+[fw,fh])*np.array(mask.shape[::-1])/[row.width,row.height]).astype(int)
        cover.append(mask[a[1]:b[1],a[0]:b[0]].mean())
    g=dict(coords=coords,fractions=fractions,MPP=mpp,physical_FOV_um=cfg["FOV_um"],native_FOV_px=[fw,fh],
           mask_tissue_mm2=tissue_mm2,FOV_mask_fraction=cover,n_candidates=qc["candidate_FOVs"],
           n_drawn=qc["selected_FOVs"],method=method,cache_sha=cache_hash,sampler_sha=contract["sampler"],
           extractor_sha=sha(SAMPLER/"extract_wsi_features.py"),source_bytes=row.bytes,
           source_mtime_ns=str(row.mtime_ns),**packed)
    temp=saved.with_suffix(".partial.npz")
    np.savez_compressed(temp,**g)
    temp.replace(saved)
    return g


def make_rebuilder(contract):
    """Call the same locked Phikon CUDA encoder, then the exact frozen Ridge."""
    sys.path.insert(0,str(STAGE/"scripts"))
    from extract_ESD_features import encode, original, BroadBridgeBundle, torch
    sys.path.insert(0,str(original.OLD/"runtime_deps"))
    from transformers import ViTModel
    assert torch.cuda.is_available() and contract["precision"]=="FP32"
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    torch.cuda.manual_seed_all(20261104)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    encoder=ViTModel.from_pretrained(original.OLD/"weights/phikon",add_pooling_layer=False,
        local_files_only=True).eval().to("cuda").requires_grad_(False)
    bridge=BroadBridgeBundle(BRIDGE)
    assert code("|".join(bridge.genes))==contract["gene_order"]
    def rebuild(images,groups,valid):
        phi,_=encode(images,encoder,contract["batch"])
        pooled=np.asarray([phi[groups==g].mean(0) for g in valid],dtype="float32")
        return bridge.gene_model.predict(np.asarray(pooled,dtype=float))
    return rebuild


def slides_for(case,slides,qc):
    rows=slides.loc[slides.Patient_ID.eq(case.Patient_ID)].join(qc[["status","source"]],on="path")
    assert rows.case_id.eq(case.case_id).all()
    rows=rows[rows.status.eq("PASS")].sort_values("filename")
    assert len(rows)==case.usable_WSI_count and len(rows)>0 and rows.path.is_unique
    return list(rows.itertuples())


def score_margin(model,values):
    x=np.atleast_2d(values).astype(float)
    s=model.estimator.decision_function(model.features.transform(x))
    return s,model.predict_proba(x)[:,1]


def analyze():
    initialize()
    lock,model,matrix,cases,contract=audit_and_select()
    sys.path.insert(0,str(SAMPLER))
    slides=pd.read_csv(OLD_WSIS,dtype={"case_id":str,"mtime_ns":str})
    qc=pd.read_csv(ESD/"ESD_WSI_extraction_QC.csv").set_index("path")
    assert qc.index.is_unique and slides.path.is_unique
    # Cache-only FOV-to-slide-to-case replay for every primary case, before any perturbations.
    replay=[]
    for case in cases.itertuples():
        rows=slides_for(case,slides,qc)
        vectors=[get_inputs(row,contract)[3]["RNA"].mean(0) for row in rows]
        value=np.asarray(vectors).mean(0)
        np.testing.assert_array_equal(value,matrix.loc[case.Patient_ID].to_numpy())
        prob=float(score_margin(model,value)[1][0])
        assert abs(prob-float(case.probability))<=1e-12
        replay.append({"anonymous_case_id":case.anonymous_case_id,"WSI_count":len(rows),
            "all_6175_features_bit_exact":True,"archived_probability":float(case.probability),
            "replayed_probability":prob,"absolute_error":abs(prob-float(case.probability))})
    table(pd.DataFrame(replay),QA/"all_59_FOV_to_patient_exact_replay.csv")
    print("All 59 patient feature vectors exactly reproduced from 749 frozen WSI caches.",flush=True)
    cache_rebuild=None
    def rebuild(images,groups,valid):
        nonlocal cache_rebuild
        if cache_rebuild is None: cache_rebuild=make_rebuilder(contract)
        return cache_rebuild(images,groups,valid)

    # Resolve all case and main-slide locks before examining any attribution values.
    selected=[]
    failure=[]
    for group in FOLDERS:
        for case in cases[cases.group.eq(group)].itertuples():
            rr=slides_for(case,slides,qc)
            geometry_rows=[]
            try:
                for i,row in enumerate(rr,1):
                    inp=get_inputs(row,contract)
                    anon_slide="WSI-"+code("stage24-slide:"+inp[0])[:10].upper()
                    g=geometry(row,inp,contract,rebuild)
                    geometry_rows.append((row,anon_slide,inp[3]["RNA"],g))
                    print(f"{group} {case.anonymous_case_id} {i}/{len(rr)} {anon_slide} coordinate verified",flush=True)
            except (openslide.OpenSlideError,ValueError) as exc:
                if isinstance(exc,ValueError) and str(exc)!="No tissue-positive patches under the frozen sampler":
                    raise
                failure.append({"group":group,"anonymous_case_id":case.anonymous_case_id,
                    "reason":type(exc).__name__,"failed_slide":anon_slide})
                write(QA/"technical_fallbacks.json",failure)
                continue
            main=min(geometry_rows,key=lambda z:(-float(z[3]["mask_tissue_mm2"]),z[1]))
            selected.append((case,geometry_rows,main[1]))
            break
        else:
            raise RuntimeError(f"No readable case in group {group}; do not substitute another classification")
    selection=[{"group":c.group,"anonymous_case_id":c.anonymous_case_id,"corrected_LNM":int(c.corrected),
                "archived_probability":float(c.probability),"selection_rank":int(c.selection_rank),
                "case_selection_reason":("Unique false negative" if c.group=="FN" else
                                         "Nearest group median; ties by anonymous ID"),
                "WSI_count":len(s),"total_valid_FOVs":sum(len(z[2]) for z in s),
                "main_slide":main,"main_slide_selection_reason":"Largest frozen tissue-support mask area; ties by anonymous WSI ID"}
               for c,s,main in selected]
    write(QA/"case_and_slide_lock_before_LOO.json",selection)
    write(QA/"technical_fallbacks.json",failure)
    table(pd.DataFrame(selection),DATA/"anonymous_case_selection.csv")
    run_loo(model,selected,matrix,lock)


def run_loo(model,selected,matrix,lock):
    reports,regions,slides,private=[],[],[],[]
    alpha=float(model.calibrator.coef_[0,0])
    beta=float(model.calibrator.intercept_[0])
    assert alpha>0
    for case,wsis,main in selected:
        slide_means=np.asarray([z[2].mean(0) for z in wsis])
        value=slide_means.mean(0)
        np.testing.assert_array_equal(value,matrix.loc[case.Patient_ID].to_numpy())
        m,p=score_margin(model,value)
        assert abs(float(p[0])-float(case.probability))<=1e-12
        assert abs(expit(alpha*m[0]+beta)-p[0])<=1e-12
        n_slides=len(wsis)
        gene_idx=model.features.genes
        coef=model.estimator.coef_[0]/model.features.scaler.scale_
        reports.append({"group":case.group,"anonymous_case_id":case.anonymous_case_id,
            "max_abs_feature_diff":0.,"archived_probability":float(case.probability),
            "full_raw_margin":float(m[0]),"replayed_probability":float(p[0]),
            "probability_error":abs(float(p[0])-float(case.probability)),
            "calibration_slope":alpha,"calibration_intercept":beta,"WSI_count":n_slides})
        for j,(row,wsid,rna,g) in enumerate(wsis):
            n=len(rna)
            fw,fh=g["native_FOV_px"]
            mx,my=g["MPP"]
            fx,fy=g["physical_FOV_um"]
            if n>1:
                perturbed=[]
                for k in range(n):
                    means=slide_means.copy()
                    means[j]=np.delete(rna,k,axis=0).mean(0)
                    perturbed.append(means.mean(0))
                m_loo,p_loo=score_margin(model,perturbed)
                dm,dp=m[0]-m_loo,p[0]-p_loo
                np.testing.assert_allclose(expit(alpha*m_loo+beta),p_loo,rtol=0,atol=2e-15)
                assert np.array_equal(np.sign(dm),np.sign(dp))
                # Independently verify the exact linear margin from the float32
                # feature deltas used at deployment. Audit the ideal float64
                # mean identity separately because averaging roundoff is real.
                actual_feature_delta=value[gene_idx].astype(float)-np.asarray(perturbed)[:,gene_idx].astype(float)
                np.testing.assert_allclose(dm,actual_feature_delta@coef,rtol=0,atol=1e-10)
                r64=rna[:,gene_idx].astype(float)
                ideal_feature_delta=(r64-r64.mean(0))/((n-1)*n_slides)
                ideal_dm=ideal_feature_delta@coef
                roundoff_dm=(actual_feature_delta-ideal_feature_delta)@coef
                np.testing.assert_allclose(dm,ideal_dm+roundoff_dm,rtol=0,atol=1e-10)
            else:
                m_loo=p_loo=dm=dp=ideal_dm=roundoff_dm=np.full(n,np.nan)
            summary=dict(group=case.group,anonymous_case_id=case.anonymous_case_id,anonymous_slide_id=wsid,
                main_slide=wsid==main,level0_width=row.width,level0_height=row.height,
                MPP_x=float(mx),MPP_y=float(my),tissue_support_area_mm2=float(g["mask_tissue_mm2"]),
                valid_FOVs=n,n_candidates=int(g["n_candidates"]),n_sampled_before_patch_filter=int(g["n_drawn"]),
                attribution_FOVs=int(np.isfinite(dm).sum()),positive_FOVs=int((dm>0).sum()),negative_FOVs=int((dm<0).sum()),
                min_margin_delta=float(np.nanmin(dm)) if n>1 else np.nan,
                max_margin_delta=float(np.nanmax(dm)) if n>1 else np.nan,
                min_FOV_support_fraction=float(min(g["FOV_mask_fraction"])),tissue_rule="Frozen clean mask, no tumor segmentation",
                coordinate_verification=str(g["method"]),quickhash_available=bool(pd.notna(row.quickhash)),
                image_QC="Source header, bounds, exact sampling checks passed")
            slides.append(summary)
            private.append(dict(anonymous_slide_id=wsid,anonymous_case_id=case.anonymous_case_id,
                internal_patient_key=row.Patient_ID,internal_accession=row.case_id,source_path=row.path,
                source_bytes=row.bytes,source_mtime_ns=str(row.mtime_ns),quickhash=row.quickhash))
            for k,(x,y) in enumerate(g["coords"]):
                patch=g["fractions"][k]
                regions.append(dict(group=case.group,anonymous_case_id=case.anonymous_case_id,anonymous_slide_id=wsid,
                    main_slide=wsid==main,FOV_ordinal=k+1,x_level0=int(x),y_level0=int(y),
                    width_level0=int(fw),height_level0=int(fh),target_width_um=float(fx),target_height_um=float(fy),
                    physical_read_width_um=float(fw*mx),physical_read_height_um=float(fh*my),
                    retained_patches=int((patch>=.05).sum()),max_patch_tissue_fraction=float(max(patch)),
                    candidate_mask_fraction=float(g["FOV_mask_fraction"][k]),full_raw_margin=float(m[0]),
                    without_FOV_raw_margin=m_loo[k],delta_raw_margin=dm[k],full_calibrated_probability=float(p[0]),
                    without_FOV_calibrated_probability=p_loo[k],delta_calibrated_probability=dp[k],
                    ideal_float64_delta_raw_margin=ideal_dm[k],float32_aggregation_roundoff_delta=roundoff_dm[k],
                    attribution_status="PASS" if n>1 else "UNDEFINED_SINGLE_FOV"))
    table(pd.DataFrame(regions),DATA/"all_WSI_FOV_attribution.csv")
    table(pd.DataFrame(slides),DATA/"anonymous_slide_QC.csv")
    table(pd.DataFrame(reports),QA/"selected_case_margin_and_probability_replay.csv")
    table(pd.DataFrame(private),PRIVATE/"slide_identity_and_source.csv")
    assert sha(Path(lock["winner_directory"])/"model.joblib")==lock["model_SHA256"]
    write(QA/"attribution_gate.json",{"status":"PASS","case_groups":len(selected),"slides":len(slides),
        "FOVs":len(regions),"raw_margin":"Frozen LR log-odds before the sigmoid calibrator",
        "calibration_slope":alpha,"model_sha256":lock["model_SHA256"],"score_threshold":lock["winner"]["thresholds"]["0.98"]})


if __name__=="__main__":
    with threadpool_limits(limits=2):
        analyze()
