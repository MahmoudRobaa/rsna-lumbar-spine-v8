import json

def code_cell(source):
    lines = source.split('\n')
    return {
        'cell_type': 'code',
        'metadata': {'trusted': True},
        'source': [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else []),
        'outputs': [],
        'execution_count': None
    }

def md_cell(source):
    lines = source.split('\n')
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    }

def create_notebook(cells):
    return {
        'metadata': {
            'kernelspec': {
                'name': 'python3',
                'display_name': 'Python 3',
                'language': 'python'
            },
            'language_info': {
                'name': 'python',
                'version': '3.12.12',
                'mimetype': 'text/x-python',
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'pygments_lexer': 'ipython3',
                'nbconvert_exporter': 'python',
                'file_extension': '.py'
            },
            'kaggle': {
                'accelerator': 'gpu',
                'dataSources': [],
                'dockerImageVersionId': 31260,
                'isInternetEnabled': True,
                'language': 'python',
                'sourceType': 'notebook',
                'isGpuEnabled': True
            }
        },
        'nbformat_minor': 4,
        'nbformat': 4,
        'cells': cells
    }
