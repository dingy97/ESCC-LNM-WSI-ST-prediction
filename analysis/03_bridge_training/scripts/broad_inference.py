"""Whole-RNA/program inference interface. No four-gene output restriction."""
from pathlib import Path
import json
import joblib
import numpy as np
from full_embedding_ridge import FullRidge

class BroadBridgeBundle:
    def __init__(self,directory):
        directory=Path(directory)
        self.card=json.loads((directory/'bundle_card.json').read_text(encoding='utf-8'))
        self.genes=np.asarray(self.card['gene_names'])
        self.gene_model=joblib.load(directory/'Phikon_FULL_B_Ridge.joblib')
        self.refinement_model=joblib.load(directory/'Phikon_FULL_C_Ridge.joblib')
        self.program_model=joblib.load(directory/'ResNet50_NMF_C.joblib')
    def predict_features(self,phikon_FOV_features,resnet_FOV_features):
        phi=np.asarray(phikon_FOV_features,dtype=float);res=np.asarray(resnet_FOV_features,dtype=float)
        if phi.ndim!=2 or res.ndim!=2 or len(phi)!=len(res):raise ValueError('Aligned FOV-level feature matrices required')
        if phi.shape[1]!=768 or res.shape[1]!=2048:raise ValueError('Expected frozen Phikon CLS 768 and ResNet50 2048 features')
        if not np.isfinite(phi).all() or not np.isfinite(res).all():raise ValueError('Nonfinite or image-QC-failed FOV')
        expression=self.gene_model.predict(phi)
        refined=self.refinement_model.predict(phi)
        program=self.program_model.latent_predict(self.program_model.feature.transform(res))
        status=np.where(self.gene_model.target.mask,'learned_target_unvalidated_gene_specific_accuracy','training_mean_fallback_failed_training_QC')
        return {'gene_names':self.genes,'gene_expression':expression,'primary_refined_gene_expression':refined,
                'gene_prediction_status':status,'program_scores':program,
                'program_names':np.asarray([f'NMF_C_{k+1}' for k in range(program.shape[1])]),
                'units':'Image-inferred FOV log1p(CPM); program scores in training-standardized units',
                'reliability':'Exploratory whole-profile representation; individual gene/program accuracy is heterogeneous and not independently validated'}
