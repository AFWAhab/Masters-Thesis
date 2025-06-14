import pandas as pd
import re
import numpy as np

# code copied from
# https://gist.github.com/rf-santos/22f521c62ca2f85ac9582bf0d91e4054

def gtf2df(gtf: str) -> pd.DataFrame:
    """
    Convert a GTF file to a pandas DataFrame.
    Tested with Ensemble GTF files.
    Parameters
    ----------
    gtf : str
        Path to the GTF file.
    Returns
    -------
    df : pandas.DataFrame
    Examples
    --------
    .. highlight:: python
    .. code-block:: python
        df = gtf2df('ensembl.gtf')
        df.head()
    """

    df = pd.read_csv(gtf, sep='\t', header=None, comment='#', low_memory=False)
    df.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    fields = ['gene_id', 'transcript_id', 'exon_number', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_name',
              'transcript_source', 'transcript_biotype', 'protein_id', 'exon_id', 'tag']
    for field in fields:
        df[field] = df['attribute'].apply(
            lambda x: re.findall(rf'{field} "([^"]*)"', x)[0] if rf'{field} "' in x else '')

    df.replace('', np.nan, inplace=True)

    df.drop('attribute', axis=1, inplace=True)

    return df