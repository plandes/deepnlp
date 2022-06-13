### Feature Vectorizer to Batch Mapping

The last bit of implementation that's needed is the binding between the domain
classes and the vectorizers.  First we define constants of the [configuration
file] entries in the [Batch] class `MovieReviewBatch` to reduce duplication in our
code:
```python
@dataclass
class ReviewBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    TRANSFORMER_EMBEDDING = 'transformer_embedding'
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING, GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING, TRANSFORMER_EMBEDDING}
    STATS_ATTRIBUTE = 'stats'
    ENUMS_ATTRIBUTE = 'enums'
    COUNTS_ATTRIBUTE = 'counts'
    DEPENDENCIES_ATTRIBUTE = 'dependencies'
    LANGUAGE_ATTRIBUTES = {STATS_ATTRIBUTE, ENUMS_ATTRIBUTE, COUNTS_ATTRIBUTE,
                           DEPENDENCIES_ATTRIBUTE}
```

Next we define the mapping of the label class and vectorizer of the label:
```python
    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'label_vectorizer_manager',
            (FieldFeatureMapping('label', 'rvlabel', True),)),
```

Now we provide the linguistic feature mapping to vectorizers, again using the
`feature_id` given in the [configuration file]:
```python
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', True, 'doc'),
              FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', True, 'doc'),
              FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_EMBEDDING, 'transformer', True, 'doc'),
              FieldFeatureMapping(STATS_ATTRIBUTE, 'stats', False, 'doc'),
              FieldFeatureMapping(ENUMS_ATTRIBUTE, 'enum', True, 'doc'),
              FieldFeatureMapping(COUNTS_ATTRIBUTE, 'count', True, 'doc'),
              FieldFeatureMapping(DEPENDENCIES_ATTRIBUTE, 'dep', True, 'doc')))
```

Finally, we return the given `MAPPINGS` class level constant in our class as
the single abstract method of the [Batch] super class:
```python
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
```
