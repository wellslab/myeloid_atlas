import numpy as np
import pd as pd
import sklearn
import sklearn.decomposition
import json

# Generate an atlas using pluripotent stem cell data
# Annotations expected to be updated in the near future

# Read in and find cut
data = pd.read_csv('<git_repo_location>/data/myeloid_atlas_expression_v7.1.tsv', sep='\t', index_col=0)
annotations = pd.read_csv('<git_repo_location>/data/myeloid_atlas_samples_v7.1.tsv', sep='\t', index_col=0)
colours = {}
for i_key in json.load(open('<git_repo_location>/myeloid_atlas_colours_v7.1.tsv'))['colours'].keys():
	colours.update(json.load(open('<git_repo_location>/myeloid_atlas_colours_v7.1.tsv'))['colours'][i_key])

data = functions.transform_to_percentile(data)
genes = functions.calculate_platform_dependence(data, annotations)

pca        = sklearn.decomposition.PCA(n_components=10, svd_solver='full')
pca.fit(functions.transform_to_percentile(data.loc[genes.Platform_VarFraction.values<=0.2]).transpose())
pca_coords = pca.transform(functions.transform_to_percentile(data.loc[genes.Platform_VarFraction.values<=0.2]).transpose())

functions.plot_pca(pca_coords, annotations,pca, \
                   labels=['Cell Type', 'Sample Source', 'Progenitor Type', 'Activation Status', 'Platform_Category', 'Disease'], colour_dict=myeloid_atlas_colours)
