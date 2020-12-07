from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.ner",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    description='An example of NER tagging using Zensols DeepNLP with the Groningen Meaning Bank corpus.',
    user='plandes',
    project='ner',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
