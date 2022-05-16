from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.deepnlp",
    package_names=['zensols', 'resources'],
    package_data={'': ['*.conf', '*.yml']},
    description='Deep learning utility library for natural language processing that aids in feature engineering and embedding layers.',
    user='plandes',
    project='deepnlp',
    keywords=['tooling'],
    has_entry_points=False,
).setup()
