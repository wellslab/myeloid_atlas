"""
This module contains classes and functions that work with the integrated expression atlas in Stemformatics.
In particular, this module is for handling calculations on the atlas, such as data projections.
You can also use this module to create the atlas using the create_atlas function.

Example usage:

    import atlas

    # In order to instantiate the Atlas object, you need the data files from the http://stemformatics.org/atlas.
    # Select the atlas type,
    atl = atlas.Atlas("/path/to/atlas/data", "blood")  # create an Atlas instance for blood atlas
    print(atl.expression().head())  # rank normalised expression matrix as pandas DataFrame

    atl.runPCA()
    print(atl.coords().head()) # show PCA coordinates after running it

    
"""
import pandas, numpy, os, json, anndata
from sklearn.decomposition import PCA
from copy import deepcopy   # For nested dictionaries, we need deepcopy, otherwise values are just references

def rankTransform(df):
    """Return a rank transformed version of data frame
    """
    return (df.shape[0] - df.rank(axis=0, ascending=False, na_option='bottom')+1)/df.shape[0]

def filepaths(pathToAtlasData, atlasType):
    """Return a dictionary of full filepaths to each key.
    Example: filenames('/Users/jarnyc/projects/atlas','blood') will return
        {"expression":"/Users/jarnyc/projects/atlas/blood_atlas_expression_v7.1.tsv", ...}
    Note that this function takes version numbers into account and will return the latest version of the file in the directory.
    """
    paths = {'expression':None, 'genes':None, 'samples':None, 'colours':None}
    allFilenames = [filename for filename in os.listdir(pathToAtlasData) if filename.startswith(atlasType)]
    for key in paths.keys():
        matchingFilenames = [filename for filename in allFilenames if key in filename]
        
        # samples may also be called annotations
        if len(matchingFilenames)==0 and key=='samples':
            matchingFilenames = [filename for filename in allFilenames if 'annotations' in filename]

        if len(matchingFilenames)==1:
            paths[key] = os.path.join(pathToAtlasData, matchingFilenames[0])
        else:
            pass # work out how to deal with multiple files matching this key (possibly due to multiple versions)
    return paths

def create_atlas(expressionDatafile, sampleDatafile):
    """Create atlas expression matrix from input files. expressionDatafile is the full path to the csv file 
    (readable into a DataFrame by pandas.read_csv function) that contains all the datasets concatenated columnwise.
    sampleDatafile is the full path to the csv file that contain all sample information. This table must contain
    a column named 'Platform_Category', which is the variable to used to filter out genes with high variance.
    Note that in the expression matrix, RNASeq data should have zeros as zeros, not nans.
    """
    # Using rpy2 package to plug in R. Also need variancePartition R package installed in this environment.
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri, rpy2.robjects.pandas2ri
    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.pandas2ri.activate()
    variancePartition = importr('variancePartition')

    # Read in expression data and sample metadata.
    data        = pandas.read_csv(expressionDatafile)
    metadata    = pandas.read_csv(sampleDatafile)
    data.dropna(how='any', inplace=True) # Drop genes that are not measurable in every dataset due to probes being absent 

    # Search for platform dependent genes 
    form              = robjects.Formula('~ Platform_Category')
    varPart           = variancePartition.fitExtractVarPartModel(transform_to_percentile(data), form, metadata[['Platform_Category']])

    sel_varPart       = numpy.array(varPart)[0] <= 0.2 #This is the filtering step
    genes_to_keep     = data.index.values[sel_varPart] #genes_to_keep is an array holding all the genes that pass the filter

    filtered_data     = rankTransform(data.loc[genes_to_keep].copy())
    return filtered_data
    

class Atlas(object):
    """Define the atlas object, which is specified by the directory where the atlas files are, and atlas type.
    Note that expression matrix will be read in at object initialisation stage.
    """
    def __init__(self, pathToAtlasData, atlasType):

        self.pathToAtlasData = pathToAtlasData
        paths = filepaths(self.pathToAtlasData, atlasType)

        genes = pandas.read_csv(paths.get('genes'), sep="\t", index_col=0)
        samples = pandas.read_csv(paths.get('samples'), sep="\t", index_col=0)
        df = pandas.read_csv(paths.get('expression'), sep="\t", index_col=0)
        colours = json.load(open(paths.get('colours')))

        # Perform some validation.
        if set(genes.index)!=set(df.index):
            raise Exception("index of genes does not match index of expression")
        elif set(samples.index)!=set(df.columns):
            raise Exception("index of samples does not match columns of expression")
        df = df[samples.index].loc[genes.index]

        # Construct instance vars - note that we re-calculate the expression values as ranks for filtered matrix
        keys = ["all","filtered","projection"]
        self.annData = {"all": anndata.AnnData(X=df.transpose(), obs=samples, var=genes)}
        self.annData["filtered"] = self.annData["all"][:,genes[genes["inclusion"]].index]
        self.annData["filtered"].X = rankTransform(self.annData["filtered"].to_df().transpose()).transpose()
        self.annData["projection"] = None  # set after projection() is run

        self.colours = dict([(key, deepcopy(colours['colours'])) for key in keys])   # {"all": {"Cell Type":{"B cell":"#cccccc", ...}, ...}, ... }
        self.ordering = dict([(key, deepcopy(colours['ordering'])) for key in keys]) # {"all": {"Cell Type":["CD8+ T cell","CD4+ T cell", ...], ...}, ...}
        self.pca = None
        self.coords = {"all":None, "filtered":None, "projection":None} # pca coords performed on expression matrix of matching key

    def expression(self, key="filtered"):
        """Return DataFrame of expression (genes x samples).
        If key="filtered", only return expression for genes included in the atlas
        """
        return self.annData[key].to_df().transpose()

    def samples(self, key="filtered"):
        return self.annData[key].obs

    def genes(self, key="filtered"):
        return self.annData[key].var

    def convertGeneSymbolsToEnsemblIds(self, df):
        """Uses the gene annotation object stored in the instance to convert row ids from gene symbols to Ensembl gene ids for supplied data frame.
            > df = atl.convertGeneSymbolsToEnsemblIds(df)
            > Shape before: (26593, 1078) , shape after: (12311, 1078)
        """
        if df.index[0].startswith("ENS"): return  # already has ensembl id
        genes = self.genes(key="all").reset_index().set_index("symbol")
        shape = df.shape
        commonSymbols = list(set(genes.index).intersection(set(df.index)))

        # drop duplicate index from expression before running reindex
        df = df.loc[~df.index.duplicated(keep='first')].reindex(commonSymbols)
        df.index = genes.reindex(commonSymbols)["ensembl"]
        print("Shape before converting to Ensembl ids:", shape, ", shape after:", df.shape)
        return df

    def runPCA(self, key="filtered"):
        """Perform PCA and save coordinates in 3d.
        """
        self.pca = PCA(n_components=10, svd_solver='full')
        df = self.pca.fit_transform(self.annData[key].X)
        self.coords[key] = pandas.DataFrame(df[:,:3], index=self.annData[key].obs_names, columns=['x','y','z'])

    def projection(self, testData, testKey="original", testPointColours="green", randomPointColours="black"):
        """Perform projection of testData onto this atlas.
        Afterwards, various object attributes are assigned:
            self.annData["projection"]: AnnData of expression where X = concatenated (columnwise) data frame of 
                self.expression("filtered") and testData.expression() after rank transform. 
                self.annData["projection"].obs will also have "projection" column added, which will have None values
                for atlas samples and testData.name for testData samples.
            self.coords["projection"]: projected coordinates as a data frame.
            self.ordering["projection"]: extends self.ordering to include test values
            self.colours["colours]: extends self.colours to include colours for test points
        """
        # Some validation before projecting
        df = self.expression()
        commonGenes = testData.expression(key=testKey).index.intersection(df.index)  # common index between test and atlas
        if not testData.sampleMap:
            raise Exception("No sampleMap, which is a required dictionary that maps samples columns of atlas to that of test data.")
        if len(commonGenes)==0:
            raise Exception("No genes common between test data and atlas, likely due to row ids not in Ensembl ids.")
        elif len(commonGenes)/len(self.genes()[self.genes()["inclusion"]])<0.5:
            raise Exception("Less than 50% of genes in test data are common with atlas ({} common)".format(len(commonGenes)))
        elif not testData.sampleMap:  # no sampleMap specified, we'll just make one up
            testData.sampleMap = {self.samples().columns[0]: testData.samples().columns[0]}
        elif testData.sampleMap:
            sm = testData.sampleMap
            if len(set(sm.keys()).intersection(set(self.samples().columns)))==0 or len(set(sm.values()).intersection(set(testData.samples().columns)))==0:
                raise Exception("SampleMap specified for test data seems incorrect.")
    
        # We reindex on df.index, not on commonGenes, since pca is done on df. This means any genes in df not found in 
        # test will gene None assigned - we will live with this, as long as there aren't so many.
        print("Projecting test onto the atlas using %s common genes" % len(commonGenes))
        dfTest = rankTransform(testData.expression(key=testKey).reindex(df.index))
        expression = pandas.concat([df, dfTest], axis=1)    # combined expression matrix (rank transformed)

        # Perform pca on atlas
        self.runPCA()
        
        # Make projection
        testCoords = pandas.DataFrame(self.pca.transform(dfTest.values.T)[:,:3], index=dfTest.columns, columns=['x','y','z'])
        self.coords["projection"] = self.coords["filtered"].append(testCoords)
        
        # Merge sample columns of test and atlas - we can only do this for columns found in testData.sampleMap
        projectionSamples = pandas.DataFrame(index=self.samples().index.tolist() + testData.samples(key=testKey).index.tolist())
        for atlasColumn,testDataColumn in testData.sampleMap.items():
            projectionSamples[atlasColumn] = self.samples()[atlasColumn].tolist() + \
                ["%s_%s" % (testData.name, testData.samples(key=testKey).at[index,testDataColumn]) for index in testData.samples(key=testKey).index]

        # Add testData to colours and ordering based on sampleMap
        for column in projectionSamples.columns:
            self.ordering["projection"][column] = deepcopy(self.ordering["filtered"][column])
            self.colours["projection"][column] = deepcopy(self.colours["filtered"][column])
            for value in testData.samples(key=testKey)[testData.sampleMap[column]].unique():
                self.ordering["projection"][column].append("%s_%s" % (testData.name,value))
                self.colours["projection"][column]["%s_%s" % (testData.name,value)] = randomPointColours if value.startswith(testData.randomPrefix) else testPointColours

        # We will also add a column to projectionSamples to keep track of which are atlas samples and which are projected ones.
        projectionSamples["projection"] = [None for index in self.samples().index] + [testData.name for index in testData.samples(key=testKey).index]

        # Add this to annData object
        self.annData["projection"] = anndata.AnnData(X=expression.transpose(), obs=projectionSamples, var=self.annData["filtered"].var)
        
        return self

    def calculateTestDataTrajectory(self, testData):
        """Once test data has been projected, those samples may lie along a trajectory and we can try to find genes whose expression are
        correlated with that trajectory.
        """
        # projected coordinates (excluding random)
        samples = self.annData["projection"].obs
        samples = samples[pandas.notnull(samples["projection"])]  # test data samples only
        df = self.coords["projection"].loc[samples.index]
        
        # Perform PCA on these coordinates and get coordinates of the first component
        pca = PCA(n_components=1, svd_solver='full')
        coords = pandas.Series(pca.fit_transform(df.values)[:,0], index=df.index)
        
        # For each gene in test, calculate correlation between these coordinates and expression values
        from scipy.stats import pearsonr
        df = testData.expression()
        
        geneCorr = []
        for index,row in df.iterrows():
            if row.var()==0: continue
            corr = pearsonr(row, coords)
            if corr[1]<0.05:
                geneCorr.append([index,corr[0],abs(corr[0])])
                
        df = pandas.DataFrame(geneCorr, columns=['geneId','corr','abs_corr']).set_index('geneId')
        df = df.sort_values('abs_corr', ascending=False)
        df["symbol"] = [self.genes().at[geneId,'symbol'] if geneId in self.genes().index else geneId for geneId in df.index]
        return df

    '''           
    def kmeans(self, k=20):
        from scipy.cluster.vq import kmeans,vq
        self.centroids,_ = kmeans(self.coords.values, k)
        clusters,_ = vq(self.coords.values, self.centroids)    # assign each sample to a cluster
        self.samples['kmeans'] = clusters

    def rankGeneGroups(self, group):
        # mean expression of each gene across samples in that group
        kmeans = self.expression().groupby(self.samples[group],axis=1).mean()
    '''

class TestData(object):
    """Read data which will be projected onto the atlas. We call this test data. Since test data may come in a variety
    of formats, we want to be more flexible with our initialisation.
    """
    
    def __init__(self, name, **kwargs):
        # Parse input
        self.name = name
        self.randomPrefix = kwargs.get('randomPrefix', 'random')
        self.sampleMap = kwargs.get('sampleMap', {})  # maps atlas columns to testData columns. {'Cell Type':'test_data_column',...}

        if 'expression' in kwargs:  # expression matrix as a pandas.DataFrame object
            df = kwargs['expression']
        else:
            df = pandas.read_csv(os.path.join(kwargs['filepath']), sep=kwargs.get('sep','\t'), index_col=kwargs.get('index_col',0))
        samples = kwargs.get('samples')

        # Validate
        if set(samples.index)!=set(df.columns):
            raise Exception("index of samples does not match columns of expression")
        df = df[samples.index]

        # Construct instance vars - remember anndata wants samples x genes
        self.annData = {'original':anndata.AnnData(X=df.transpose(), obs=samples)}

    def expression(self, key="original"):
        return self.annData[key].to_df().transpose()

    def samples(self, key="original"):
        return self.annData[key].obs

    def addRandom(self, n=3, key="original", prefix="random"): 
        """Add some random values, based on distribution of values already present in this dataset.
        Parameters
            n (int): total number of random values added.
            key (str): one of ['original','aggregated',...] - which expression matrix should be used for value distribution and where random values 
                should be added.
            prefix (str): used to add a prefix for sample ids of random values so that they can be identified.
        """
        import random
        self.randomPrefix = prefix

        # Target expression matrix we will work with (genes x samples)
        df = self.annData[key].to_df().transpose()

        # n columns are randomly chosen, then for each column we shuffle the values
        n = min([n, len(df.columns)])

        # Create a new AnnData object of random values and concatenate with existing annData
        randomDf = pandas.DataFrame(index=df.index)
        randomSamples = pandas.DataFrame(columns=self.annData[key].obs.columns)
        for i,col in enumerate(random.sample(df.columns.to_list(), n)):
            randomDf["%s%i" % (self.randomPrefix,i)] = df[col].reset_index(drop=True).sample(frac=1).to_list()
            randomSamples.loc["%s%i" % (self.randomPrefix,i)] = [self.randomPrefix for item in randomSamples.columns]
        self.annData[key] = self.annData[key].concatenate([anndata.AnnData(X=randomDf.transpose(), obs=randomSamples)], index_unique=None)

    def aggregateSamples(self, from_key="original", to_key="aggregated", sampleGroup=None, n=3):
        """For single cell dataset, we can aggregate values based on membership in sampleGroup 
        (which must correspond to a column of self.annData[from_key].obs). Creates a self.annData[to_key] object 
        where obs has sampleGroup as a column.
            > testData.aggregateSamples(sampleGroup="celltype")
            > print(testData.expression("celltype").shape)
        n (int): how many points to represent the smallest cluster with
        """
        import random
        if sampleGroup is None:
            sampleGroup = self.samples(key=from_key).columns[0]
        elif sampleGroup not in self.samples(key=from_key).columns:
            raise Exception("'%s' is not found in annData['%s'].obs.columns" % (sampleGroup, from_key))

        valueCounts = self.samples(key=from_key)[sampleGroup].value_counts().sort_values()
        smallest = valueCounts.tolist()[0] # size of the smallest cluster
        
        if smallest<=3*n: # if the smallest cluster isn't at least 3 times the value of n, just use all of that cluster
            n = 1
        sampleSize = int(smallest/n)  # how many samples to put into each aggregated cluster
        if sampleSize==1:  # no aggregation happening
            raise Exception("Could not determine how many samples to aggregate together. Likely due to a cluster having only one member.")
        print("sample size for each cluster:", sampleSize)

        aggregated = pandas.DataFrame(index=self.expression(key=from_key).index)
        samples = pandas.DataFrame(columns=[sampleGroup])
        for item in valueCounts.index:  # each item of sampleGroup
            df = self.expression(key=from_key)[self.samples(key=from_key)[self.samples(key=from_key)[sampleGroup]==item].index]  # expression for this sampleGroup
            # we need new sample ids based on sample group eg. "Mono1__0"
            i = 0
            columns = df.columns.tolist()
            while len(columns)>sampleSize:
                selectedColumns = random.sample(columns, sampleSize)
                aggregated["%s__%s" % (item, i)] = df[selectedColumns].sum(axis=1)
                samples.at["%s__%s" % (item, i), sampleGroup] = item
                columns = set(columns).difference(set(selectedColumns))
                i += 1

        self.annData[to_key] = anndata.AnnData(X=aggregated.transpose(), obs=samples, var=self.annData[from_key].var)
        return self


################################################################################
# For testing. Eg: nosetests -s atlas.py:test_atlas
################################################################################
def test_filepaths():
    print(filepaths("/Users/jarnyc/projects/BloodAtlas/received/Stemformatics", "blood"))
    print(filepaths("/Users/jarnyc/projects/BloodAtlas/received/Stemformatics", "myeloid"))

def test_atlas():
    atl = Atlas("/Users/jarnyc/projects/BloodAtlas/received/Stemformatics", "blood")
    assert atl.expression().shape[0] > 3700
    assert len([item for item in atl.expression().columns if '7379' in item])>10

def test_addRandom():
    df = pandas.read_csv("/Users/jarnyc/projects/BloodAtlas/received/Galen/GSE116256_RAW/GSM3587996_BM1.dem.txt.gz", 
                         sep="\t", index_col=0, compression='gzip')
    samples = pandas.read_csv("/Users/jarnyc/projects/BloodAtlas/received/Galen/GSE116256_RAW/GSM3587996_BM1.anno.txt.gz", 
                                sep="\t", index_col=0, compression='gzip')
    test = TestData("galen", expression=df, samples=samples)
    test.addRandom()
    assert len([item for item in test.expression().columns if item.startswith(test.randomPrefix)])>0
    assert len([item for item in test.expression()["random0"] if pandas.notnull(item)])==len(test.expression())

def test_convertGeneSymbolsToEnsemblIds():
    atl = Atlas("/Users/jarnyc/projects/BloodAtlas/received/Stemformatics", "blood")


def test_projection():
    atl = Atlas("/Users/jarnyc/projects/BloodAtlas/received/Stemformatics", "blood")
    df = pandas.read_csv("/Users/jarnyc/projects/BloodAtlas/received/Haemopedia/Haemopedia-Human-RNASeq_tpm.txt", sep="\t", index_col=0)
    samples = pandas.read_csv("/Users/jarnyc/projects/BloodAtlas/received/Haemopedia/Haemopedia-Human-RNASeq_samples.txt", sep="\t", index_col=0)
    test = TestData("haem", expression=df, samples=samples, sampleMap = {"Cell Type":"celltype"})
    print(atl.samples().shape, test.samples().shape, atl.projection(test).samples("projection").shape)
    print(len(atl.ordering["filtered"]["Cell Type"]), len(atl.ordering["projection"]["Cell Type"]))

def test_trajectory():    
    atl = Atlas("/Users/jarnyc/projects/BloodAtlas/received/Stemformatics", "blood")
    df = pandas.read_csv("/Users/jarnyc/projects/BloodAtlas/received/Haemopedia/Haemopedia-Human-RNASeq_tpm.txt", sep="\t", index_col=0)
    samples = pandas.read_csv("/Users/jarnyc/projects/BloodAtlas/received/Haemopedia/Haemopedia-Human-RNASeq_samples.txt", sep="\t", index_col=0)
    test = TestData("haem", expression=df, samples=samples, sampleMap = {"Cell Type":"celltype"})
    print(atl.projection(test).calculateTestDataTrajectory(test).head())
