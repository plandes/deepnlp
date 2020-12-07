from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="movie",
    package_names=['movie', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    description='The Stanford/Cornell movie review dataset to demonstrate the DeepZenos framework for the sentiment analysis task.',
    user='plandes',
    project='movie',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
