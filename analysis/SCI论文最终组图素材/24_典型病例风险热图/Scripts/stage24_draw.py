"""Draw exactly four independently selected H&E examples from verified Stage24 sources."""
from contextlib import closing
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import openslide

from stage24_attribution import ROOT, OUT, DATA, QA, PRIVATE, FOLDERS, SAMPLER, check_quickhash, sha, write, table
sys.path.insert(0,str(ROOT/"23_manuscript_figures/figure2/scripts"))
from style import MM, INK, MUTED, ORANGE, plt
sys.path.insert(0,str(SAMPLER))
from spatial_sampling import slide_candidate_mask
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm, ListedColormap, to_rgb, to_hex
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

ROI_EDGE="#F4B13D"
FOV_EDGE="#146A85"
NEGATIVE_COLOR, POSITIVE_COLOR = "#14559F", "#BD3B35"
PANELS=("Original_HE","FOV_locations","LNM_attribution_heatmap","HE_ROI_magnification")


def read_csv(p, **kw):
    return pd.read_csv(p,float_precision="round_trip",**kw)


def footprint(r):
    return r.x_level0,r.y_level0,r.width_level0,r.height_level0


def tissue_path(mask, scale, box):
    """Clip display color to the same candidate tissue support, inside each actual FOV."""
    x,y,w,h=box
    sx,sy=scale
    i,j=np.floor([x/sx,y/sy]).astype(int)
    right,bottom=np.ceil([(x+w)/sx,(y+h)/sy]).astype(int)
    binary=mask[j:bottom,i:right].astype(np.uint8)
    contours,_=cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    paths=[]
    for con in contours:
        con=cv2.approxPolyDP(con,.32,True).reshape(-1,2)
        if len(con)<3: continue
        xy=(con+[i+.5,j+.5])*[sx,sy]
        paths.append(MplPath(np.r_[xy,xy[:1]],closed=True))
    if not paths: return None
    return MplPath.make_compound_path(*paths)


def crop_overview(slide, mpp, df):
    dims=np.asarray(slide.dimensions)
    small=np.asarray(slide.get_thumbnail((2048,2048)).convert("RGB"))
    mask,_=slide_candidate_mask(small,slide.dimensions,mpp)
    ms=dims/np.asarray(mask.shape[::-1])
    sy,sx=np.nonzero(mask)
    assert len(sy)>0
    left=min(sx.min()*ms[0],df.x_level0.min())
    top=min(sy.min()*ms[1],df.y_level0.min())
    right=max((sx.max()+1)*ms[0],(df.x_level0+df.width_level0).max())
    bottom=max((sy.max()+1)*ms[1],(df.y_level0+df.height_level0).max())
    margin=450/np.array(mpp)
    thumb=slide.get_thumbnail((4200,4200)).convert("RGB")
    ts=dims/np.asarray(thumb.size)
    a=np.maximum(0,np.floor((np.array([left,top])-margin)/ts).astype(int))
    b=np.minimum(thumb.size,np.ceil((np.array([right,bottom])+margin)/ts).astype(int))
    crop=thumb.crop((*a,*b))
    extent=(a[0]*ts[0],b[0]*ts[0],b[1]*ts[1],a[1]*ts[1])
    assert left>=extent[0] and right<=extent[1] and top>=extent[3] and bottom<=extent[2]
    return crop,extent,mask,ms


def layout(title,note,image,extent,mpp,scale_um):
    fig=plt.figure(figsize=(114*MM,103*MM))
    fig.text(.036,.968,title,va="top",fontsize=9,fontweight="bold")
    fig.text(.036,.925,note,va="top",fontsize=7,color=MUTED)
    # Match physical aspect even on WSI with small differences between X and Y MPP.
    ratio=(extent[1]-extent[0])*mpp[0]/((extent[2]-extent[3])*mpp[1])
    bw,bh=.93,.735
    w=min(bw,bh*103*ratio/114)
    h=w*114/(103*ratio)
    x,y=.5-w/2,.129+(.735-h)/2
    ax=fig.add_axes((x,y,w,h))
    ax.imshow(np.asarray(image),extent=extent,interpolation="none",aspect="auto")
    ax.set(xlim=extent[:2],ylim=extent[2:])
    ax.set_axis_off()
    # The bar is in the footer. Its length uses exactly the image panel's X transform.
    bar_width=w*scale_um/((extent[1]-extent[0])*mpp[0])
    x0=.95-bar_width
    y0=.097
    fig.add_artist(Line2D([x0,.95],[y0,y0],transform=fig.transFigure,color=INK,lw=1.5,solid_capstyle="butt"))
    fig.text(.95,y0-.015,"1 mm" if scale_um==1000 else f"{scale_um} µm",ha="right",va="top",fontsize=6.6)
    return fig,ax


def box(ax,r,color,lw=.75,ls="-"):
    x,y,w,h=footprint(r)
    shape=Rectangle((x,y),w,h,fill=False,edgecolor=color,linewidth=lw,linestyle=ls,zorder=6)
    ax.add_patch(shape)
    return shape


def export(fig,folder,name):
    for ext in ("pdf","svg","png"):
        fig.savefig(folder/f"{name}.{ext}",dpi=600)
    plt.close(fig)


def main(heatmap_only=False):
    assert json.loads((QA/"attribution_gate.json").read_text(encoding="utf-8"))["status"]=="PASS"
    selection=read_csv(DATA/"anonymous_case_selection.csv")
    regions=read_csv(DATA/"all_WSI_FOV_attribution.csv")
    slides=read_csv(DATA/"anonymous_slide_QC.csv")
    private=read_csv(PRIVATE/"slide_identity_and_source.csv",dtype={"source_mtime_ns":str}).set_index("anonymous_slide_id")
    max_val=regions.delta_raw_margin.abs().max()
    assert np.isfinite(max_val) and max_val>0
    # Ceiling to three decimal places for a readable, symmetric, non-clipping scale.
    bound=float(np.ceil(max_val*1000)/1000)
    norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    def overlay(value):
        return to_rgb(POSITIVE_COLOR if value>0 else NEGATIVE_COLOR),.76*np.sqrt(abs(value)/bound)
    # Show the common opacity encoding on a white reference in the color bar.
    palette=[]
    for value in np.linspace(-bound,bound,257):
        rgb,alpha=overlay(value)
        palette.append((1-alpha)+alpha*np.asarray(rgb))
    cmap=ListedColormap(palette)
    fig_qc=[]
    roi_rows=[]
    for case in selection.itertuples():
        d=regions[regions.anonymous_slide_id.eq(case.main_slide)]
        s=slides.loc[slides.anonymous_slide_id.eq(case.main_slide)].iloc[0]
        link=private.loc[case.main_slide]
        folder=OUT/FOLDERS[case.group]
        assert not d.empty and d.main_slide.all() and d.anonymous_case_id.eq(case.anonymous_case_id).all()
        assert len(d)==s.valid_FOVs and d.attribution_status.eq("PASS").all()
        roi=d.assign(abs_delta=d.delta_raw_margin.abs()).sort_values(["abs_delta","FOV_ordinal"],ascending=[False,True]).iloc[0]
        fp=Path(link.source_path)
        stat=fp.stat()
        assert str(stat.st_mtime_ns)==link.source_mtime_ns and stat.st_size==link.source_bytes
        mpp=(s.MPP_x,s.MPP_y)
        with closing(openslide.OpenSlide(str(fp))) as slide:
            assert slide.dimensions==(s.level0_width,s.level0_height)
            check_quickhash(slide,link.quickhash)
            he,extent,mask,scale=crop_overview(slide,mpp,d)
            x,y,w,h=footprint(roi)
            detail=slide.read_region((int(x),int(y)),0,(int(w),int(h))).convert("RGB")
        note=f"{case.anonymous_case_id}  |  {case.group}  |  Frozen patient probability {case.archived_probability:.3f}"
        if not heatmap_only:
            fig,ax=layout("Original H&E",note,he,extent,mpp,1000)
            export(fig,folder,PANELS[0])
            fig,ax=layout("Sampled FOV footprints",note,he,extent,mpp,1000)
            for r in d.itertuples(): box(ax,r,FOV_EDGE,.75)
            box(ax,roi,ROI_EDGE,1.3,"--")
            fig.text(.038,.092,f"{len(d)} valid FOVs on the displayed WSI",fontsize=6.7,color=MUTED)
            fig.text(.038,.060,"Dashed amber: magnified FOV",fontsize=6.7,color=INK)
            export(fig,folder,PANELS[1])

        fig,ax=layout("LNM risk attribution",note,he,extent,mpp,1000)
        drawn=0
        for r in d.itertuples():
            if r.delta_raw_margin==0: continue
            path=tissue_path(mask,scale,footprint(r))
            if path is None:
                raise RuntimeError(f"Missing display tissue support in {r.anonymous_slide_id} FOV {r.FOV_ordinal}")
            x,y,w,h=footprint(r)
            rgb,alpha=overlay(r.delta_raw_margin)
            patch=PathPatch(path,facecolor=rgb,edgecolor="none",alpha=alpha,zorder=4)
            ax.add_patch(patch)
            clip=Rectangle((x,y),w,h,transform=ax.transData)
            patch.set_clip_path(clip)
            drawn+=1
        box(ax,roi,ROI_EDGE,1.3,"--")
        cax=fig.add_axes((.045,.087,.37,.021))
        sm=ScalarMappable(norm=norm,cmap=cmap)
        cb=fig.colorbar(sm,cax=cax,orientation="horizontal",ticks=[-bound,0,bound])
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.ax.set_xticklabels([f"{-bound:.3f}","0",f"+{bound:.3f}"])
        cb.ax.tick_params(labelsize=6.3)
        fig.text(.045,.030,"Full minus leave-one-FOV-out LR log-odds",fontsize=6.5,color=INK)
        export(fig,folder,PANELS[2])

        if not heatmap_only:
            fig,ax=layout("Native-resolution H&E detail",note,detail,(0,w,h,0),mpp,100)
            fig.text(.038,.091,f"FOV {int(roi.FOV_ordinal)}  |  Δ log-odds {roi.delta_raw_margin:+.4f}",fontsize=6.7,color=INK)
            fig.text(.038,.061,"Largest absolute attribution on the selected WSI",fontsize=6.7,color=MUTED)
            export(fig,folder,PANELS[3])

        approx_area=np.sum(d.candidate_mask_fraction*d.physical_read_width_um*d.physical_read_height_um)/1e6
        fig_qc.append({"group":case.group,"anonymous_case_id":case.anonymous_case_id,
                       "anonymous_slide_id":case.main_slide,"overview_crop_level0_left":extent[0],
                       "overview_crop_level0_top":extent[3],"overview_crop_level0_right":extent[1],
                       "overview_crop_level0_bottom":extent[2],"overview_pixels_x":he.width,
                       "overview_pixels_y":he.height,"ROI_native_pixels_x":detail.width,"ROI_native_pixels_y":detail.height,
                       "valid_FOVs":len(d),"drawn_tissue_polygons":drawn,"overview_bar_um":1000,"ROI_bar_um":100,
                       "tissue_area_mm2":s.tissue_support_area_mm2,"approx_sampled_tissue_area_mm2":approx_area,
                       "approx_sampled_tissue_fraction":approx_area/s.tissue_support_area_mm2,
                       "ROI_color_on_white":to_hex(cmap(norm(roi.delta_raw_margin))),"global_color_scale":bound})
        roi_rows.append(roi.drop(labels=["abs_delta"]).to_dict())
    table(pd.DataFrame(fig_qc),QA/"figure_geometry_and_coverage.csv")
    table(pd.DataFrame(roi_rows),DATA/"displayed_ROI.csv")
    write(QA/"plotting_contract.json",{"raw_margin_delta_absolute_limit":bound,"max_observed_absolute_delta":max_val,
        "scope":"All FOVs on all included WSI of four selected accession groups",
        "color_alpha":"0.76 * sqrt(abs(delta)/global_scale_limit); colorbar shows that opacity on white",
        "positive":"red","negative":"blue","FOV_color_display":"Mask-intersected virtual FOV footprint only",
        "ROI":"Native Level 0 region, exact virtual FOV","image_interpolation_for_colors":"none",
        "svg_font":"editable text","png_dpi":600,
        "attribution_csv_sha256":sha(DATA/"all_WSI_FOV_attribution.csv")})


if __name__=="__main__":
    main(heatmap_only="--heatmap-only" in sys.argv[1:])
