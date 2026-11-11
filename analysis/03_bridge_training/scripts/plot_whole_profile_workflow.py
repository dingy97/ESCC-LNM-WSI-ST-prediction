from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import fitz
from PIL import Image
import train_bridge as tr
R=tr.R;O=R/'06_figures/bridge_figures'
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.size':10})
fig,ax=plt.subplots(figsize=(11,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
boxes=[(.02,.69,.26,.20,'H&E FOV bags\n375 FOVs / 26 patients\nNative physical patch scale','#EAF3F8'),
       (.36,.69,.27,.20,'Frozen image encoders\nPhikon + ResNet50\nFOV-level feature pooling','#EAF3F8'),
       (.73,.69,.25,.20,'Patient-nested validation\nSame 24 primary patients\nTrain-only preprocessing','#EAF3F8'),
       (.08,.32,.34,.21,'Whole RNA expression output\n6,175 gene columns retained\nQC/accuracy status per gene','#E7F2EB'),
       (.57,.32,.34,.21,'Molecular-program output\nTraining-derived PCA/NMF/modules\nFull-panel gene loadings retained','#E7F2EB'),
       (.19,.01,.63,.16,'Future clinical ML comparison (not run in stage two)\nFull inferred RNA / programs / joint representations / pure WSI\nPatient-level selection and validation; preserve the 22 shared identities','#F0F0F0')]
for x,y,w,h,label,color in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',facecolor=color,edgecolor='#4A4A4A',lw=1))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.5)
for a,b in [((.29,.79),(.34,.79)),((.64,.79),(.71,.79)),((.50,.67),(.25,.55)),((.50,.67),(.74,.55)),((.25,.30),(.40,.19)),((.74,.30),(.60,.19))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#333333','lw':1.3})
fig.suptitle('Broad transcriptome and molecular-program prediction',fontsize=12)
fig.tight_layout();name='SupplementaryS5_whole_transcriptome_ML_workflow'
for ext in ['pdf','svg','png']:fig.savefig(O/f'{name}.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
caption='The deliverable preserves the full RNA panel and training-derived molecular programs. Per-gene predictability is an accuracy diagnostic, not a four-gene output restriction or an assumed LNM mechanism. Clinical classification and WSI-wide inference are future work and were not run in stage two.'
p=O/'figure_captions.csv';a=pd.read_csv(p);a=pd.concat([a,pd.DataFrame([{'figure':name,'caption':caption}])]).drop_duplicates('figure',keep='last');a.to_csv(p,index=False)
(O/'图注.md').write_text('# 第二阶段图注\n\n'+'\n\n'.join('## '+r.figure+'\n\n'+r.caption for r in a.itertuples()),encoding='utf-8')
qa=[];thumbs=[];out=O/'rendered_QA'
for pdf in sorted(O.glob('*.pdf')):
    with fitz.open(pdf) as doc:
        page=doc[0];assert len(doc)==1 and page.get_text().strip()
        image=out/(pdf.stem+'.png');page.get_pixmap(matrix=fitz.Matrix(1.1,1.1)).save(image)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox'];assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1
        qa.append({'pdf':pdf.name,'pages':1,'fonts':len(page.get_fonts()),'vector_drawings':len(page.get_drawings()),'text_bounds_ok':True,'rendered':True})
        im=Image.open(image).convert('RGB');im.thumbnail((780,520));thumbs.append(im)
for batch in range((len(thumbs)+3)//4):
    canvas=Image.new('RGB',(1560,1040),'white')
    for i,im in enumerate(thumbs[batch*4:batch*4+4]):canvas.paste(im,((i%2)*780,(i//2)*520))
    canvas.save(out/f'contact_{batch+1}.jpg',quality=94)
tr.dump(O/'PDF_structural_QA.json',qa)
print('Whole-transcriptome workflow diagram exported; all 13 figure PDFs rendered')
