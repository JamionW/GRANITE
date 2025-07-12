"""
Warning suppression module - import this FIRST before any other GRANITE modules
"""
import warnings
import os
import sys

def suppress_all_r_warnings():
    """Suppress R and rpy2 warnings before they can appear"""
    
    # Suppress all rpy2 warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='rpy2')
    warnings.filterwarnings('ignore', message='.*Environment variable.*redefined by R.*')
    warnings.filterwarnings('ignore', message='.*libraries.*contain no packages.*')
    
    # Set R environment variables to prevent warnings
    os.environ['R_LIBS_SITE'] = '/usr/local/lib/R/site-library:/usr/lib/R/site-library:/usr/lib/R/library'
    os.environ['R_LIBS_USER'] = ''
    
    # Suppress Python warnings related to R
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

# Call it immediately when this module is imported
suppress_all_r_warnings()